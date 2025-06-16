# Add all necessary imports at the top
import torch
import numpy as np
from torch.distributions import Categorical
from .models import ActorSoftmax, Critic
from .buffer import PGReplay
from .min_norm_solver import MinNormSolver, cagrad_exact
import threading
import queue
import time

def func(grads_list):
    sol, min_norm = MinNormSolver.find_min_norm_element_FW(grads_list)
    scale = [float(sol[i]) for i in range(len(grads_list))] # +1 reference
    scaled_grad = torch.zeros_like(grads_list[0])  # Adjusted for proper tensor initialization
    for i in range(len(scale)):
        scaled_grad += grads_list[i] * scale[i]
    return scaled_grad
    
    
def run_with_timeout(target, args, timeout, ref_grad):
    result_queue = queue.Queue()
    
    def wrapper():
        result = target(args)
        result_queue.put(result)
    
    thread = threading.Thread(target=wrapper)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print("Function timed out and was skipped")
        return -ref_grad  # Return negative reference gradient if timeout occurs
    return result_queue.get() if not result_queue.empty() else None


def get_obj_grad_kl(ref_vec_KL, returns):
    returns = returns/(returns.norm(p=1,dim=-1).reshape(-1,1)+0.0001)
    grad_0 = ref_vec_KL[0]/returns[:,0]
    grad_1 = ref_vec_KL[1]/returns[:,1]
    return torch.cat([grad_0.reshape(-1,1),grad_1.reshape(-1,1)],dim=1)
    
class Agent:
    def __init__(self,cfg) -> None:
        
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device)
        self.actor = ActorSoftmax(cfg.n_states+cfg.r_dim,cfg.n_actions, hidden_dim = cfg.actor_hidden_dim).to(self.device)
        self.critic = Critic(cfg.n_states+cfg.r_dim,cfg.r_dim,hidden_dim=cfg.critic_hidden_dim).to(self.device) # mo dim
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PGReplay()
        self.k_epochs = cfg.k_epochs # update policy for K epochs
        self.eps_clip = cfg.eps_clip # clip parameter for PPO
        self.entropy_coef = cfg.entropy_coef # entropy coefficient
        self.sample_count = 0
        self.update_freq = cfg.update_freq
        self.ref_diff = 0
        self.ref_adv = 0
        self.lam = 2
        self.r_dim = cfg.r_dim
        self.method = cfg.MO_algo_name
        self.method_list = {"CAP":self.update_CAP,"cagrad":self.update_CAGrad, "PreCo":self.update_PreCo,"EPO":self.update_EPO,"LS":self.update_LS,"SDMgrad":self.update_sdmg}
        assert self.method in self.method_list,'choose from "cagrad","PreCo","EPO","LS","SDMgrad","CAP"'

    def sample_action(self,state):
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
      
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs = dist.log_prob(action).detach()
        return action.detach().cpu().numpy().item()
    @torch.no_grad()
    def predict_action(self,state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy().item()

    @torch.no_grad()        
    def greedy_action(self,state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
     
        return probs.argmax(dim=-1).detach().cpu().numpy().item()
    
    def update(self,ref_vec):
        self.method_list[self.method](ref_vec)
    
    def update_LS(self, ref_vec):
        self.lam += 0.00002
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        # print("update policy")
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()
        # convert to tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0

        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)

        #returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        for kk in range(self.k_epochs):
            # compute advantage
            values = self.critic(old_states) # detach to avoid backprop through the critic
            advantage = returns - values.detach()
            # get action probabilities
            probs = self.actor(old_states)
            dist = Categorical(probs)
            # get new action probabilities
            new_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_probs - old_log_probs) # old_log_probs must be detached

            r_dim = advantage.shape[-1]
            b_dim = advantage.shape[0]
            ratio = ratio.repeat(r_dim).reshape(r_dim,b_dim).transpose(0,-1)
       
            logprob_grad = ratio.detach()*advantage.detach()
         
            logprob_grad[torch.where(ratio>1 + self.eps_clip)] = 0
            logprob_grad[torch.where(ratio<1 - self.eps_clip)] = 0
            grads_list = [-logprob_grad[:,i] for i in range(r_dim)]
            #ref_vec = torch.rand(r_dim)#torch.tensor([0.8,0.2,0,0,0,0])#torch.rand(r_dim)

            # reference vector
            ref_vec = torch.tensor(ref_vec, device=self.device, dtype=torch.float32)
            ref_vec_KL = ref_vec/ref_vec.norm(p=1)
            # ref_vec = ref_vec/ref_vec.norm()
            obj_vec = returns

            obj_grad_KL = get_obj_grad_kl(ref_vec_KL,returns)# ref_1/obj_1, ref_2/obj_2


            #print(obj_vec[-5:],'nanan')
            obj_vec = obj_vec/(obj_vec.norm(dim=-1).reshape(obj_vec.shape[0],1)+0.0001)
            objj = obj_vec.mean(0)
            objj = objj/objj.norm()
            delta_obj = ref_vec.cuda()-obj_vec.detach()

            #print("asfasfsf",delta_obj.norm(dim=-1).sum()>(ref_vec.cuda()-objj).norm())


            self.ref_diff = (delta_obj.norm(dim=-1)).mean().item()
            asobj = returns/(returns.norm(p=1,dim=-1).reshape(-1,1)+0.0001)+0.0001
            loggg = torch.log(ref_vec_KL/asobj)
            loggg[torch.isnan(loggg)]=0
            loggg[loggg==-np.inf]=0
            loggg[loggg==np.inf]=0


            self.ref_diff = torch.sum(ref_vec_KL*loggg,dim=-1)

            self.ref_diff = self.ref_diff.mean()




            delta_obj = torch.clamp(delta_obj.mean(dim=0),-0.2,999)
            obj_grad = delta_obj/delta_obj.norm()

           

            delta_obj = torch.clamp(obj_grad_KL.mean(dim=0),0,999)

 

            self.ref_adv = obj_vec.detach().cpu().numpy()
            ref_grad = (logprob_grad.detach()*obj_grad.unsqueeze(0)).sum(-1)
           
            '''for grad in grads_list:
                grad -= self.lam*ref_grad'''
            
            st = time.time()
         
            loss_grad_list = []
            for i in range(r_dim):
                g = torch.zeros_like(obj_grad)
                g[i]=-1
                loss_grad_list.append(g)
            loss_grad_list.append(-obj_grad)
   
            scaled_grad = torch.zeros_like(logprob_grad[:,0])

            for i in range(len(grads_list)):
                    scaled_grad += grads_list[i]*ref_vec[i]
            
   
            


            # compute actor loss
            #actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            actor_loss = scaled_grad.dot(new_probs)/b_dim + self.entropy_coef * dist.entropy().mean()
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()
        
    def update_CAP(self, ref_vec):
        self.lam += 0.00002
        CAPmag = 1
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        # print("update policy")
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()
        # convert to tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        probs = self.actor(old_states)
        dist = Categorical(probs)
        entropy = dist.entropy().detach()
        reversed_ent = torch.flip(entropy,[0])
      
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0
        
        for reward, done,ent in zip(reversed(old_rewards), reversed(old_dones),reversed_ent):
            if done:
                discounted_sum = 0
        
            discounted_sum = reward+(CAPmag*self.entropy_coef*ent).item() + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)

        #returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        for kk in range(self.k_epochs):
            # compute advantage
            values = self.critic(old_states) # detach to avoid backprop through the critic
            advantage = returns - values.detach()
            # get action probabilities
            probs = self.actor(old_states)
            dist = Categorical(probs)
            # get new action probabilities
            new_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_probs - old_log_probs) # old_log_probs must be detached

            r_dim = advantage.shape[-1]
            b_dim = advantage.shape[0]
            ratio = ratio.repeat(r_dim).reshape(r_dim,b_dim).transpose(0,-1)
       
            logprob_grad = ratio.detach()*advantage.detach()
         
            logprob_grad[torch.where(ratio>1 + self.eps_clip)] = 0
            logprob_grad[torch.where(ratio<1 - self.eps_clip)] = 0
            grads_list = [-logprob_grad[:,i] for i in range(r_dim)]
            #ref_vec = torch.rand(r_dim)#torch.tensor([0.8,0.2,0,0,0,0])#torch.rand(r_dim)

            # reference vector
            ref_vec = torch.tensor(ref_vec, device=self.device, dtype=torch.float32)
            ref_vec_KL = ref_vec/ref_vec.norm(p=1)
            # ref_vec = ref_vec/ref_vec.norm()
            obj_vec = returns

            obj_grad_KL = get_obj_grad_kl(ref_vec_KL,returns)# ref_1/obj_1, ref_2/obj_2


            #print(obj_vec[-5:],'nanan')
            obj_vec = obj_vec/(obj_vec.norm(dim=-1).reshape(obj_vec.shape[0],1)+0.0001)
            objj = obj_vec.mean(0)
            objj = objj/objj.norm()
            delta_obj = ref_vec.cuda()-obj_vec.detach()

            #print("asfasfsf",delta_obj.norm(dim=-1).sum()>(ref_vec.cuda()-objj).norm())


            self.ref_diff = (delta_obj.norm(dim=-1)).mean().item()
            asobj = returns/(returns.norm(p=1,dim=-1).reshape(-1,1)+0.0001)+0.0001
            loggg = torch.log(ref_vec_KL/asobj)
            loggg[torch.isnan(loggg)]=0
            loggg[loggg==-np.inf]=0
            loggg[loggg==np.inf]=0


            self.ref_diff = torch.sum(ref_vec_KL*loggg,dim=-1)

            self.ref_diff = self.ref_diff.mean()




            delta_obj = torch.clamp(delta_obj.mean(dim=0),-0.2,999)
            obj_grad = delta_obj/delta_obj.norm()

           

            delta_obj = torch.clamp(obj_grad_KL.mean(dim=0),0,999)

 

            self.ref_adv = obj_vec.detach().cpu().numpy()
            ref_grad = (logprob_grad.detach()*obj_grad.unsqueeze(0)).sum(-1)
           
            '''for grad in grads_list:
                grad -= self.lam*ref_grad'''
            
            st = time.time()
         
            loss_grad_list = []
            for i in range(r_dim):
                g = torch.zeros_like(obj_grad)
                g[i]=-1
                loss_grad_list.append(g)
            loss_grad_list.append(-obj_grad)
   
            scaled_grad = torch.zeros_like(logprob_grad[:,0])

            for i in range(len(grads_list)):
                    scaled_grad += grads_list[i]*ref_vec[i]
            
   
            


            # compute actor loss
            #actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            actor_loss = scaled_grad.dot(new_probs)/b_dim + CAPmag*self.entropy_coef * dist.entropy().mean()
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()        
    def update_sdmg(self, ref_vec):
        self.lam += 0.0013
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        # print("update policy")
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()
        # convert to tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0

        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)

        #returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        for kk in range(self.k_epochs):
            # compute advantage
            values = self.critic(old_states) # detach to avoid backprop through the critic
            advantage = returns - values.detach()
            # get action probabilities
            probs = self.actor(old_states)
            dist = Categorical(probs)
            # get new action probabilities
            new_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_probs - old_log_probs) # old_log_probs must be detached

            r_dim = advantage.shape[-1]
            b_dim = advantage.shape[0]
            ratio = ratio.repeat(r_dim).reshape(r_dim,b_dim).transpose(0,-1)
         
            logprob_grad = ratio.detach()*advantage.detach()
           
            logprob_grad[torch.where(ratio>1 + self.eps_clip)] = 0
            logprob_grad[torch.where(ratio<1 - self.eps_clip)] = 0
            grads_list = [-logprob_grad[:,i] for i in range(r_dim)]
            #ref_vec = torch.rand(r_dim)#torch.tensor([0.8,0.2,0,0,0,0])#torch.rand(r_dim)

            # reference vector
            ref_vec = torch.tensor(ref_vec, device=self.device, dtype=torch.float32)
            ref_vec_KL = ref_vec/ref_vec.norm(p=1)
            # ref_vec = ref_vec/ref_vec.norm()
            obj_vec = returns

            obj_grad_KL = get_obj_grad_kl(ref_vec_KL,returns)# ref_1/obj_1, ref_2/obj_2


            #print(obj_vec[-5:],'nanan')
            obj_vec = obj_vec/(obj_vec.norm(dim=-1).reshape(obj_vec.shape[0],1)+0.0001)
            objj = obj_vec.mean(0)
            objj = objj/objj.norm()
            delta_obj = ref_vec.cuda()-obj_vec.detach()

            #print("asfasfsf",delta_obj.norm(dim=-1).sum()>(ref_vec.cuda()-objj).norm())


            self.ref_diff = (delta_obj.norm(dim=-1)).mean().item()
            asobj = returns/(returns.norm(p=1,dim=-1).reshape(-1,1)+0.0001)+0.0001
            loggg = torch.log(ref_vec_KL/asobj)
            loggg[torch.isnan(loggg)]=0
            loggg[loggg==-np.inf]=0
            loggg[loggg==np.inf]=0


            #asobj[torch.where(asobj==0)]=ref_vec_KL.repeat(asobj.shape[0],1)[torch.where(asobj==0)]
            self.ref_diff = torch.sum(ref_vec_KL*loggg,dim=-1)

            self.ref_diff = self.ref_diff.mean()




            delta_obj = torch.clamp(delta_obj.mean(dim=0),-0.2,999)
            obj_grad = delta_obj/delta_obj.norm()

           

            delta_obj = torch.clamp(obj_grad_KL.mean(dim=0),0,999)



            self.ref_adv = obj_vec.detach().cpu().numpy()
            ref_grad = (logprob_grad.detach()*obj_grad.unsqueeze(0)).sum(-1)
           
            for grad in grads_list:
                grad -= self.lam*ref_grad
            
            st = time.time()
            #print("vecdiff",self.ref_diff,obj_grad)
            loss_grad_list = []
            for i in range(r_dim):
                g = torch.zeros_like(obj_grad)
                g[i]=-1
                loss_grad_list.append(g)
            loss_grad_list.append(-obj_grad)
         
            
            try:
                sol, min_norm = MinNormSolver.find_min_norm_element_FW(grads_list)
                scale = [float(sol[i]) for i in range(len(grads_list))] # +1 reference
                scaled_grad = torch.zeros_like(logprob_grad[:,0])
                for i in range(len(scale)):
                    scaled_grad += grads_list[i]*scale[i]
            except:
                scaled_grad = -ref_grad
            


            # compute actor loss
            #actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            actor_loss = scaled_grad.dot(new_probs)/b_dim + self.entropy_coef * dist.entropy().mean()
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()
    def update_EPO(self, ref_vec):
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        # print("update policy")
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()
        # convert to tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0

        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)

        #returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        for kk in range(self.k_epochs):
            # compute advantage
            values = self.critic(old_states) # detach to avoid backprop through the critic
            advantage = returns - values.detach()
            # get action probabilities
            probs = self.actor(old_states)
            dist = Categorical(probs)
            # get new action probabilities
            new_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_probs - old_log_probs) # old_log_probs must be detached

            r_dim = advantage.shape[-1]
            b_dim = advantage.shape[0]
            ratio = ratio.repeat(r_dim).reshape(r_dim,b_dim).transpose(0,-1)

            logprob_grad = ratio.detach()*advantage.detach()
       
            logprob_grad[torch.where(ratio>1 + self.eps_clip)] = 0
            logprob_grad[torch.where(ratio<1 - self.eps_clip)] = 0
            grads_list = [-logprob_grad[:,i] for i in range(r_dim)]
            #ref_vec = torch.rand(r_dim)#torch.tensor([0.8,0.2,0,0,0,0])#torch.rand(r_dim)

            # reference vector
            ref_vec = torch.tensor(ref_vec, device=self.device, dtype=torch.float32)
            ref_vec_KL = ref_vec/ref_vec.norm(p=1)
            # ref_vec = ref_vec/ref_vec.norm()
            obj_vec = returns

            obj_grad_KL = get_obj_grad_kl(ref_vec_KL,returns)# ref_1/obj_1, ref_2/obj_2


            #print(obj_vec[-5:],'nanan')
            obj_vec = obj_vec/(obj_vec.norm(dim=-1).reshape(obj_vec.shape[0],1)+0.0001)
            objj = obj_vec.mean(0)
            objj = objj/objj.norm()
            delta_obj = ref_vec.cuda()-obj_vec.detach()

            #print("asfasfsf",delta_obj.norm(dim=-1).sum()>(ref_vec.cuda()-objj).norm())


            self.ref_diff = (delta_obj.norm(dim=-1)).mean().item()
            asobj = returns/(returns.norm(p=1,dim=-1).reshape(-1,1)+0.0001)+0.0001
            loggg = torch.log(ref_vec_KL/asobj)
            loggg[torch.isnan(loggg)]=0
            loggg[loggg==-np.inf]=0
            loggg[loggg==np.inf]=0


            #asobj[torch.where(asobj==0)]=ref_vec_KL.repeat(asobj.shape[0],1)[torch.where(asobj==0)]
            self.ref_diff = torch.sum(ref_vec_KL*loggg,dim=-1)

            self.ref_diff = self.ref_diff.mean()




            delta_obj = torch.clamp(delta_obj.mean(dim=0),-0.2,999)
            obj_grad = delta_obj/delta_obj.norm()

           

            delta_obj = torch.clamp(obj_grad_KL.mean(dim=0),0,999)


            self.ref_adv = obj_vec.detach().cpu().numpy()
            ref_grad = (logprob_grad.detach()*obj_grad.unsqueeze(0)).sum(-1)
            
            #grads_list.append(-ref_grad)
            st = time.time()
            #print("vecdiff",self.ref_diff,obj_grad)
            loss_grad_list = []
            for i in range(r_dim):
                g = torch.zeros_like(obj_grad)
                g[i]=-1
                loss_grad_list.append(g)
            loss_grad_list.append(-obj_grad)
 
            if self.ref_diff>0.0003:
                scaled_grad = -ref_grad
                
            else:
                print("within constraint")
                
                try:
                    t0 = time.time()
                    sol, min_norm = MinNormSolver.find_min_norm_element_FW(grads_list)
                    scale = [float(sol[i]) for i in range(len(grads_list))] # +1 reference
                    scaled_grad = torch.zeros_like(logprob_grad[:,0])
                    for i in range(len(scale)):
                        scaled_grad += grads_list[i]*scale[i]
                except: 
                    scaled_grad = -ref_grad




            #surr1 = ratio * advantage[:,0]
            #surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage[:,0]


            # compute actor loss
            #actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            actor_loss = scaled_grad.dot(new_probs)/b_dim + self.entropy_coef * dist.entropy().mean()
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()
 
    def update_PreCo(self, ref_vec):
        if self.lam<=5:
            self.lam += 0.0013 # for 5-6 0.0013   last: 1-14
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        # print("update policy")
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()
        # convert to tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0

        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
      
        returns = torch.tensor(np.array(returns), device=self.device, dtype=torch.float32)

        #returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        for kk in range(self.k_epochs):
            # compute advantage
            values = self.critic(old_states) # detach to avoid backprop through the critic
            advantage = returns - values.detach()
            # get action probabilities
            probs = self.actor(old_states)
            dist = Categorical(probs)
            # get new action probabilities
            new_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_probs - old_log_probs) # old_log_probs must be detached

            r_dim = advantage.shape[-1]
            b_dim = advantage.shape[0]
            ratio = ratio.repeat(r_dim).reshape(r_dim,b_dim).transpose(0,-1)
           
            logprob_grad = ratio.detach()*advantage.detach()
    
            logprob_grad[torch.where(ratio>1 + self.eps_clip)] = 0
            logprob_grad[torch.where(ratio<1 - self.eps_clip)] = 0
            grads_list = [-logprob_grad[:,i] for i in range(r_dim)]
            #ref_vec = torch.rand(r_dim)#torch.tensor([0.8,0.2,0,0,0,0])#torch.rand(r_dim)

            # reference vector
            ref_vec = torch.tensor(ref_vec, device=self.device, dtype=torch.float32)
            ref_vec_KL = ref_vec/ref_vec.norm(p=1)
            # ref_vec = ref_vec/ref_vec.norm()
            obj_vec = returns

            obj_grad_KL = get_obj_grad_kl(ref_vec_KL,returns)# ref_1/obj_1, ref_2/obj_2


            #print(obj_vec[-5:],'nanan')
            obj_vec = obj_vec/(obj_vec.norm(dim=-1).reshape(obj_vec.shape[0],1)+0.0001)
            objj = obj_vec.mean(0)
            objj = objj/objj.norm()
            delta_obj = ref_vec.cuda()-obj_vec.detach()

            #print("asfasfsf",delta_obj.norm(dim=-1).sum()>(ref_vec.cuda()-objj).norm())


            self.ref_diff = (delta_obj.norm(dim=-1)).mean().item()
            asobj = returns/(returns.norm(p=1,dim=-1).reshape(-1,1)+0.0001)+0.0001
            loggg = torch.log(ref_vec_KL/asobj)
            loggg[torch.isnan(loggg)]=0
            loggg[loggg==-np.inf]=0
            loggg[loggg==np.inf]=0


            #asobj[torch.where(asobj==0)]=ref_vec_KL.repeat(asobj.shape[0],1)[torch.where(asobj==0)]
            self.ref_diff = torch.sum(ref_vec_KL*loggg,dim=-1)

            self.ref_diff = self.ref_diff.mean()




            delta_obj = torch.clamp(delta_obj.mean(dim=0),-0.2,999)
            obj_grad = delta_obj/delta_obj.norm()

           

            delta_obj = torch.clamp(obj_grad_KL.mean(dim=0),0,999)


            self.ref_adv = obj_vec.detach().cpu().numpy()
            ref_grad = (logprob_grad.detach()*obj_grad.unsqueeze(0)).sum(-1)
            
            for grad in grads_list:
                grad -= self.lam*ref_grad
                grad /= (1+self.lam)
          
          
            
            st = time.time()
            #print("vecdiff",self.ref_diff,obj_grad)
            loss_grad_list = []
            for i in range(r_dim):
                g = torch.zeros_like(obj_grad)
                g[i]=-1
                loss_grad_list.append(g)
            loss_grad_list.append(-obj_grad)
    
            scaled_grad = run_with_timeout(func, grads_list, 125, ref_grad)
       
            '''try:
                sol, min_norm = MinNormSolver.find_min_norm_element_FW(grads_list)
                scale = [float(sol[i]) for i in range(len(grads_list))] # +1 reference
                scaled_grad = torch.zeros_like(logprob_grad[:,0])
                for i in range(len(scale)):
                    scaled_grad += grads_list[i]*scale[i]
            except:
                print("skip minnorm")
                scaled_grad = -ref_grad'''
     



            # compute actor loss
            #actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            actor_loss = scaled_grad.dot(new_probs)/b_dim + self.entropy_coef * dist.entropy().mean()
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()

    
    def update_CAGrad(self, ref_vec):
        self.lam += 0.0003
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        # print("update policy")
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()
        # convert to tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0

        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)

        #returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        for kk in range(self.k_epochs):
            # compute advantage
            values = self.critic(old_states) # detach to avoid backprop through the critic
            advantage = returns - values.detach()
            # get action probabilities
            probs = self.actor(old_states)
            dist = Categorical(probs)
            # get new action probabilities
            new_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_probs - old_log_probs) # old_log_probs must be detached

            r_dim = advantage.shape[-1]
            b_dim = advantage.shape[0]
            ratio = ratio.repeat(r_dim).reshape(r_dim,b_dim).transpose(0,-1)
            logprob_grad = ratio.detach()*advantage.detach()
         
            logprob_grad[torch.where(ratio>1 + self.eps_clip)] = 0
            logprob_grad[torch.where(ratio<1 - self.eps_clip)] = 0
            grads_list = [-logprob_grad[:,i] for i in range(r_dim)]
            #ref_vec = torch.rand(r_dim)#torch.tensor([0.8,0.2,0,0,0,0])#torch.rand(r_dim)

            # reference vector
            ref_vec = torch.tensor(ref_vec, device=self.device, dtype=torch.float32)
            ref_vec_KL = ref_vec/ref_vec.norm(p=1)
            # ref_vec = ref_vec/ref_vec.norm()
            obj_vec = returns

            obj_grad_KL = get_obj_grad_kl(ref_vec_KL,returns)# ref_1/obj_1, ref_2/obj_2


            #print(obj_vec[-5:],'nanan')
            obj_vec = obj_vec/(obj_vec.norm(dim=-1).reshape(obj_vec.shape[0],1)+0.0001)
            objj = obj_vec.mean(0)
            objj = objj/objj.norm()
            delta_obj = ref_vec.cuda()-obj_vec.detach()

            self.ref_diff = (delta_obj.norm(dim=-1)).mean().item()
            asobj = returns/(returns.norm(p=1,dim=-1).reshape(-1,1)+0.0001)+0.0001
            loggg = torch.log(ref_vec_KL/asobj)
            loggg[torch.isnan(loggg)]=0
            loggg[loggg==-np.inf]=0
            loggg[loggg==np.inf]=0


            #asobj[torch.where(asobj==0)]=ref_vec_KL.repeat(asobj.shape[0],1)[torch.where(asobj==0)]
            self.ref_diff = torch.sum(ref_vec_KL*loggg,dim=-1)

            self.ref_diff = self.ref_diff.mean()




            delta_obj = torch.clamp(delta_obj.mean(dim=0),-0.2,999)
            obj_grad = delta_obj/delta_obj.norm()

           

            delta_obj = torch.clamp(obj_grad_KL.mean(dim=0),0,999)

    #        obj_grad = delta_obj/delta_obj.norm()
#             mag=1
#             if delta_obj.norm()<mag:
#                 obj_grad = mag*delta_obj/(delta_obj.norm()+0.00001)
#             else:
#                 obj_grad = delta_obj

            #print(obj_grad,"OBj_GRAD")

            self.ref_adv = obj_vec.detach().cpu().numpy()
            ref_grad = (logprob_grad.detach()*obj_grad.unsqueeze(0)).sum(-1)
           
            for grad in grads_list:
                grad -= self.lam*ref_grad
            
            st = time.time()
            #print("vecdiff",self.ref_diff,obj_grad)
            loss_grad_list = []
            for i in range(r_dim):
                g = torch.zeros_like(obj_grad)
                g[i]=-1
                loss_grad_list.append(g)
            loss_grad_list.append(-obj_grad)
            #print(obj_grad,"lGL",delta_obj)

            # sol_loss, _ = MinNormSolver.find_min_norm_element_FW(loss_grad_list)
            # loss_scale = [float(sol_loss[i]) for i in range(len(loss_grad_list))]
       
            # min norm
            #print(len(grads_list),grads_list[0].shape,"asda")
            try:
                scaled_grad = cagrad_exact(torch.stack(grads_list), self.r_dim, ref_grad)
                # sol, min_norm = MinNormSolver.find_min_norm_element_FW(grads_list)
                # scale = [float(sol[i]) for i in range(len(grads_list))] # +1 reference
                # scaled_grad = torch.zeros_like(logprob_grad[:,0])
                # for i in range(len(scale)):
                #     scaled_grad += grads_list[i]*scale[i]
            except:
                scaled_grad = -ref_grad
            
#             if scale[0]==0:
#                 print(ref_grad[:5],grads_list[0][:5],"sss",obj_grad)

#             if np.random.rand(1)>0.8:
#                 scale = [1,0,0,0]
            #scale[0]+=1
            #scale = [0.9,0.1,0,0,0,0]



            #surr1 = ratio * advantage[:,0]
            #surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage[:,0]


            # compute actor loss
            #actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            actor_loss = scaled_grad.dot(new_probs)/b_dim + self.entropy_coef * dist.entropy().mean()
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()

