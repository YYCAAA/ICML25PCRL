import torch
import numpy as np
from utils.test import test, test4, test5, test6
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pygmo import hypervolume
import copy
testfs = {3: test, 4: test4, 5: test5, 6: test6}
def compute_sparsity(obj_batch):
    non_dom = NonDominatedSorting().do(obj_batch, only_non_dominated_front=True)
    objs = obj_batch[non_dom]
    ONGVR = np.round(len(non_dom)/len(obj_batch),2)
    sparsity_sum = 0
    for objective in range(objs.shape[-1]):
        objs_sort = np.sort(objs[:,objective])
        sp = 0
        for i in range(len(objs_sort)-1):
            sp +=  np.power(objs_sort[i] - objs_sort[i+1],2)
        sparsity_sum += sp
    if len(objs) > 1:
        sparsity = sparsity_sum/(len(objs)-1)
    else:
        sparsity = 0
    return sparsity, ONGVR

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0)) 
    
def train(cfg, env, agent):
    print("Starting ... ")
    rewards = []  
    steps = []
    best_ep_reward = 0 
    output_agent = None
    ref_vec_list = []
    RS = []
    Hr_l = []
    Hv = []
    NRs = []
    SPs = []
    
    for j in range(cfg.ref_train_eps):
        if j%10==0:
            print(j,"p samples")
        if j%100==0:
            if j>2000:
                res_dic,mean_rs,refs = testfs[cfg.r_dim](cfg, env, agent)
                ref_point = np.zeros(cfg.r_dim)
                RS.append(np.array(mean_rs))
                hn = mean_rs/np.linalg.norm(mean_rs,axis=1).reshape(-1,1)
                Hr = np.diag(np.matmul(hn,np.array(refs).T))
                Hr_l.append((j,Hr.mean()))
                SP, NR = compute_sparsity(-np.array(mean_rs))
                print("SP",SP,"NR",NR)
                NRs.append(NR)
                SPs.append(SP)
                hvfast = hypervolume(-np.array(mean_rs))
                v = hvfast.compute(np.zeros(cfg.r_dim))
                # v = hv.hypervolume(-np.array(mean_rs), ref_point) # deap too slow for 6D
                Hv.append(v)
                print("HV:",Hv,"HR:",Hr.mean())
                print(cfg.seed,"seed",Hr_l)
#         ref_vec = np.zeros(cfg.r_dim)
#         ref_vec[np.random.randint(cfg.r_dim)] = 1
        
        # 2-D:
        # ref_ang = np.random.rand(cfg.r_dim-1)*np.pi/2
        # ref_vec = np.array([np.sin(ref_ang),np.cos(ref_ang)]).reshape(-1)
        
        # Higher-D
        while True:
            ref_vec = np.random.multivariate_normal(np.zeros(cfg.r_dim),np.eye(cfg.r_dim))
            if all(ref_vec>0):
                break
        ref_vec /= np.linalg.norm(ref_vec, 2)
       
        ref_vec_list.append(ref_vec)
        for i_ep in range(cfg.train_eps):
            ep_reward = 0  
            ep_step = 0
            state = env.reset()[0]  
            for _ in range(cfg.max_steps):
                ep_step += 1
                state = np.concatenate([state,ref_vec])
                action = agent.sample_action(state)  
                next_state, reward, done, _ , _= env.step(action)  
                #reward = reward[1:]
                reward = reward[:cfg.r_dim]
                agent.memory.push((state, action,agent.log_probs,reward,done)) 
                state = next_state  
                agent.update(ref_vec) 
                ep_reward += reward 
                if done:
                    break
            if (i_ep+1)%cfg.eval_per_episode == 0:
                sum_eval_reward = 0
                for _ in range(cfg.eval_eps):
                    eval_ep_reward = 0
                    state = env.reset()[0]
                    for _ in range(cfg.max_steps):
                        state = np.concatenate([state,ref_vec])
                        action = agent.predict_action(state) 
                        next_state, reward, done, _ , _ = env.step(action) 
                        #reward = reward[1:]
                        reward = reward[:cfg.r_dim]
                        state = next_state  
                        eval_ep_reward += reward  
                        if done:
                            break
                    sum_eval_reward += eval_ep_reward
                mean_eval_reward = sum_eval_reward/cfg.eval_eps
                
            steps.append(ep_step)
            rewards.append(ep_reward)
        
        
     
    print("done!!!!!!!!")
    output_agent = copy.deepcopy(agent) # last agent
    env.close()
    return output_agent,{'rewards':rewards,'ref_vec_list':ref_vec_list},{'Hr':Hr_l,'Rs':RS,'Hv':Hv,'SP':SPs,'NR':NRs}

def train_reacher(cfg, env, agent):
    print("Starting ... ")
    rewards = []  
    steps = []
    best_ep_reward = 0 
    output_agent = None
    ref_vec_list = []
    RS = []
    Hr_l = []
    Hv = []
    NRs = []
    SPs = []
    
    for j in range(cfg.ref_train_eps):
        if j%1==0:
            print(j,"p samples")
        if j>= 200:
            agent.entropy_coef/=100
        if j>= 300:
            agent.entropy_coef/=10
        if j%25==0:
            if j>20:
                res_dic,mean_rs,refs = testfs[cfg.r_dim](cfg, env, agent)
                ref_point = np.zeros(cfg.r_dim)
                RS.append(np.array(mean_rs))
                hn = mean_rs/np.linalg.norm(mean_rs,axis=1).reshape(-1,1)
                rn = refs/np.linalg.norm(refs,axis=1).reshape(-1,1)
                Hr = np.diag(np.matmul(hn,np.array(refs).T))
                CS = np.diag(np.matmul(hn,np.array(rn).T))
                Hr_l.append((j,Hr.mean()))
                SP, NR = compute_sparsity(-np.array(mean_rs))
                print("SP",SP,"NR",NR,"CS",CS.mean())
                NRs.append(NR)
                SPs.append(SP)
                
                hvfast = hypervolume(np.clip(-np.array(mean_rs),-999,100))
                v = hvfast.compute(100*np.ones(cfg.r_dim))
                # v = hv.hypervolume(-np.array(mean_rs), ref_point) # deap too slow for 6D
                Hv.append(v)
                print("HV:",Hv,"HR:",Hr.mean())
                print(cfg.seed,"seed",Hr_l)
#         ref_vec = np.zeros(cfg.r_dim)
#         ref_vec[np.random.randint(cfg.r_dim)] = 1
        
        # 2-D:
        # ref_ang = np.random.rand(cfg.r_dim-1)*np.pi/2
        # ref_vec = np.array([np.sin(ref_ang),np.cos(ref_ang)]).reshape(-1)
        
        # Higher-D
        while True:
            ref_vec = np.random.multivariate_normal(np.zeros(cfg.r_dim),np.eye(cfg.r_dim))
            if all(ref_vec>0):
                break
        ref_vec /= np.linalg.norm(ref_vec, 2)
       
        ref_vec_list.append(ref_vec)
        for i_ep in range(cfg.train_eps):
            ep_reward = 0  
            ep_step = 0
            state = env.reset()[0]  
          
            for t_ in range(cfg.max_steps):
                ep_step += 1
                state = np.concatenate([state,ref_vec])
                action = agent.sample_action(state)  
                next_state, reward, done, _ , _= env.step(action)  
                #reward = reward[1:]
                reward = reward[:cfg.r_dim]
                if t_ == cfg.max_steps-1:
                    done = True
                agent.memory.push((state, action,agent.log_probs,reward,done)) 
                state = next_state  
                agent.update(ref_vec) 
                ep_reward += reward 
                
                if done:
                    
                    break
            if (i_ep+1)%cfg.eval_per_episode == 0:
                sum_eval_reward = 0
                for _ in range(1):
                    eval_ep_reward = 0
                    state = env.reset()[0]
                    for _ in range(cfg.max_steps):
                        state = np.concatenate([state,ref_vec])
                        action = agent.greedy_action(state) 
                        next_state, reward, done, _ , _ = env.step(action) 
                        #reward = reward[1:]
                        reward = reward[:cfg.r_dim]
                        state = next_state  
                        eval_ep_reward += reward  
                        if done:
                            break
                        
                    sum_eval_reward += eval_ep_reward
                mean_eval_reward = sum_eval_reward
                print(mean_eval_reward,ref_vec)
                
            steps.append(ep_step)
            rewards.append(ep_reward)
        
        
     
    print("done!!!!!!!!")
    output_agent = copy.deepcopy(agent) # last agent
    env.close()
    return output_agent,{'rewards':rewards,'ref_vec_list':ref_vec_list},{'Hr':Hr_l,'Rs':RS,'Hv':Hv,'SP':SPs,'NR':NRs}
