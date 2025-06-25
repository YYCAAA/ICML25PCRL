"""PreCo off-policy implementation."""

import os
from typing import List, Optional, Union
from typing_extensions import override
from morl_baselines.common.pareto import filter_pareto_dominated
from morl_baselines.common.performance_indicators import hypervolume
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import random
import copy
from sklearn.metrics.pairwise import cosine_similarity
from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.evaluation import (
    log_all_multi_policy_metrics,
    log_episode_info,
)
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.networks import (
    NatureCNN,
    get_grad_norm,
    layer_init,
    mlp,
    polyak_update,
)
from morl_baselines.common.prioritized_buffer import PrioritizedReplayBuffer
from morl_baselines.common.utils import linearly_decaying_value
from morl_baselines.common.weights import equally_spaced_weights, random_weights,w_test6,w_test4

class Qmem:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()

    def append(self, critic):
        if self.capacity > 0:
            self._append(critic)
    def _append(self, critic):
        if len(self.buffer) < self.capacity:
            co = copy.deepcopy(critic)
            self.buffer.append(co)
        else:
            self.buffer[self._p] = critic
        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def reset(self):
        self._n = 0
        self._p = 0
        self.full = False
        self.buffer = []
    def sample(self):
        return self.buffer

class w_Adaptor(nn.Module):
    """Network for adapting the Preco weight"""
    def __init__(self, rew_dim, net_arch = [16,16]):
        """Initialize the Q network.

        Args:
            obs_shape: shape of the observation
            action_dim: number of actions
            rew_dim: number of objectives
            net_arch: network architecture (number of units per layer)
        """
        super().__init__()
        self.w_dim = rew_dim
        self.bl = 1/rew_dim
        self.net = mlp(self.w_dim, self.w_dim, net_arch)
        self.apply(layer_init)
    
    def forward(self, w):
        out = th.softmax(self.net(w), dim=-1)-self.bl
        return out+w
        
class QNet(nn.Module):
    """Multi-objective Q-Network conditioned on the weight vector."""

    def __init__(self, obs_shape, action_dim, rew_dim, net_arch):
        """Initialize the Q network.

        Args:
            obs_shape: shape of the observation
            action_dim: number of actions
            rew_dim: number of objectives
            net_arch: network architecture (number of units per layer)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rew_dim = rew_dim
        if len(obs_shape) == 1:
            self.feature_extractor = None
            input_dim = obs_shape[0] + rew_dim
        elif len(obs_shape) > 1:  # Image observation
            self.feature_extractor = NatureCNN(self.obs_shape, features_dim=512)
            input_dim = self.feature_extractor.features_dim + rew_dim
        # |S| + |R| -> ... -> |A| * |R|
        self.net = mlp(input_dim, action_dim * rew_dim, net_arch)
        self.apply(layer_init)
       

    def forward(self, obs, w):
        """Predict Q values for all actions.

        Args:
            obs: current observation
            w: weight vector

        Returns: the Q values for all actions

        """
        if self.feature_extractor is not None:
            features = self.feature_extractor(obs)
            if w.dim() == 1:
                w = w.unsqueeze(0)
            input = th.cat((features, w), dim=features.dim() - 1)
        else:
            input = th.cat((obs, w), dim=w.dim() - 1)
        q_values = self.net(input)
        return q_values.view(-1, self.action_dim, self.rew_dim)  # Batch size X Actions X Rewards


def batched_min_norm_solver(G, max_iter=250):
    """
    G: Tensor of shape (B, M, K) — B problems, M-dim gradients, K tasks
    Returns:
        w: Tensor of shape (B, K)
        agg_grad: Tensor of shape (B, M)
    """
    B, M, K = G.shape
    G_T = G.transpose(1, 2)  # shape: (B, K, M)
    GG = th.bmm(G_T, G)   # (B, K, K): Gram matrix

    w = th.zeros(B, K, device=G.device)
    w[:, 0] = 1.0  # init at a vertex of the simplex

    for t in range(1, max_iter + 1):
        grad = 2 * th.bmm(GG, w.unsqueeze(-1)).squeeze(-1)  # (B, K)
        s_idx = grad.argmin(dim=-1)  # index of best direction (B,)
        s = th.zeros_like(w)
        s[th.arange(B), s_idx] = 1.0

        gamma = 2.0 / (t + 2)
        w = w + gamma * (s - w)

    agg_grad = th.bmm(G, w.unsqueeze(-1)).squeeze(-1)  # (B, M)
    return w, agg_grad

class ANet(nn.Module):
    """Multi-objective Q-Network conditioned on the weight vector."""

    def __init__(self, obs_shape, action_dim, rew_dim, net_arch):
        """Initialize the Q network.

        Args:
            obs_shape: shape of the observation
            action_dim: number of actions
            rew_dim: number of objectives
            net_arch: network architecture (number of units per layer)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rew_dim = rew_dim
        if len(obs_shape) == 1:
            self.feature_extractor = None
            input_dim = obs_shape[0] + rew_dim
        elif len(obs_shape) > 1:  # Image observation
            self.feature_extractor = NatureCNN(self.obs_shape, features_dim=512)
            input_dim = self.feature_extractor.features_dim + rew_dim
        # |S| + |R| -> ... -> |A|
        self.net = mlp(input_dim, action_dim, net_arch)
        self.apply(layer_init)
       

    def forward(self, obs, w):
        """Predict Q values for all actions.

        Args:
            obs: current observation
            w: weight vector

        Returns: the Q values for all actions

        """
        if self.feature_extractor is not None:
            features = self.feature_extractor(obs)
            if w.dim() == 1:
                w = w.unsqueeze(0)
            input = th.cat((features, w), dim=features.dim() - 1)
        else:
            input = th.cat((obs, w), dim=w.dim() - 1)
        q_values = self.net(input)
        return q_values.view(-1, self.action_dim)  # Batch size X Actions X Rewards



class PreCo(MOPolicy, MOAgent):
 

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        initial_epsilon: float = 0.01,
        final_epsilon: float = 0.01,
        epsilon_decay_steps: int = None,  # None == fixed epsilon
        tau: float = 1.0,
        target_net_update_freq: int = 200,  # ignored if tau != 1.0
        buffer_size: int = int(1e6),
        net_arch: List = [256, 256, 256, 256],
        batch_size: int = 256,
        learning_starts: int = 100,
        gradient_updates: int = 1,
        gamma: float = 0.99,
        max_grad_norm: Optional[float] = 1.0,
        envelope: bool = True,
        num_sample_w: int = 4,
        per: bool = True,
        per_alpha: float = 0.6,
        initial_homotopy_lambda: float = 0.0,
        final_homotopy_lambda: float = 1.0,
        homotopy_decay_steps: int = None,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "Envelope",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = 10,
        device: Union[th.device, str] = "auto",
        group: Optional[str] = None,
    ):
        """Envelope Q-learning algorithm.

        Args:
            env: The environment to learn from.
            learning_rate: The learning rate (alpha).
            initial_epsilon: The initial epsilon value for epsilon-greedy exploration.
            final_epsilon: The final epsilon value for epsilon-greedy exploration.
            epsilon_decay_steps: The number of steps to decay epsilon over.
            tau: The soft update coefficient (keep in [0, 1]).
            target_net_update_freq: The frequency with which the target network is updated.
            buffer_size: The size of the replay buffer.
            net_arch: The size of the hidden layers of the value net.
            batch_size: The size of the batch to sample from the replay buffer.
            learning_starts: The number of steps before learning starts i.e. the agent will be random until learning starts.
            gradient_updates: The number of gradient updates per step.
            gamma: The discount factor (gamma).
            max_grad_norm: The maximum norm for the gradient clipping. If None, no gradient clipping is applied.
          
            per: Whether to use prioritized experience replay.
            per_alpha: The alpha parameter for prioritized experience replay.
            initial_homotopy_lambda: The initial value of the homotopy parameter for homotopy optimization.
            final_homotopy_lambda: The final value of the homotopy parameter.
            homotopy_decay_steps: The number of steps to decay the homotopy parameter over.
            project_name: The name of the project, for wandb logging.
            experiment_name: The name of the experiment, for wandb logging.
            wandb_entity: The entity of the project, for wandb logging.
            log: Whether to log to wandb.
            seed: The seed for the random number generator.
            device: The device to use for training.
            group: The wandb group to use for logging.
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device)
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.per = per
        self.per_alpha = per_alpha
        self.gradient_updates = gradient_updates
        self.initial_homotopy_lambda = initial_homotopy_lambda
        self.final_homotopy_lambda = final_homotopy_lambda
        self.homotopy_decay_steps = homotopy_decay_steps
        self.q_net = QNet(self.observation_shape, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
        self.a_net = ANet(self.observation_shape, self.action_dim, self.reward_dim, net_arch=[64,64]).to(self.device)
        self.target_q_net = QNet(self.observation_shape, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
        self.target_a_net = ANet(self.observation_shape, self.action_dim, self.reward_dim, net_arch=[64,64]).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_a_net.load_state_dict(self.a_net.state_dict())
        #self.w_net = w_Adaptor(self.reward_dim).to(self.device)
        self.lam = 10
        self.exp = 8
        self.Qmem = Qmem(4)
        
        seed = 8
        
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        self.seed = seed
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = True
        for param in self.target_q_net.parameters():
            param.requires_grad = False

        self.q_optim = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.a_optim = optim.Adam(self.a_net.parameters(), lr=self.learning_rate)
        #self.w_optim = optim.Adam(self.w_net.parameters(), lr=self.learning_rate)
     
        self.num_sample_w = num_sample_w
        self.homotopy_lambda = self.initial_homotopy_lambda
        print(self.seed,"seed")
        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.observation_shape,
                1,
                rew_dim=self.reward_dim,
                max_size=buffer_size,
                action_dtype=np.uint8,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.observation_shape,
                1,
                rew_dim=self.reward_dim,
                max_size=buffer_size,
                action_dtype=np.uint8,
            )

        self.log = log
        if log:
            self.setup_wandb(project_name, experiment_name, wandb_entity, group)

    @override
    def get_config(self):
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "clip_grand_norm": self.max_grad_norm,
            "target_net_update_freq": self.target_net_update_freq,
            "gamma": self.gamma,
            "use_envelope": 0,
            "num_sample_w": self.num_sample_w,
            "net_arch": self.net_arch,
            "per": self.per,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.buffer_size,
            "initial_homotopy_lambda": self.initial_homotopy_lambda,
            "final_homotopy_lambda": self.final_homotopy_lambda,
            "homotopy_decay_steps": self.homotopy_decay_steps,
            "learning_starts": self.learning_starts,
            "seed": self.seed,
        }

    def save(self, save_replay_buffer: bool = True, save_dir: str = "weights/", filename: Optional[str] = None):
        """Save the model and the replay buffer if specified.

        Args:
            save_replay_buffer: Whether to save the replay buffer too.
            save_dir: Directory to save the model.
            filename: filename to save the model.
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        saved_params = {}
        saved_params["q_net_state_dict"] = self.q_net.state_dict()

        saved_params["q_net_optimizer_state_dict"] = self.q_optim.state_dict()
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.experiment_name if filename is None else filename
        th.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path: str, load_replay_buffer: bool = True):
        """Load the model and the replay buffer if specified.

        Args:
            path: Path to the model.
            load_replay_buffer: Whether to load the replay buffer too.
        """
        params = th.load(path)
        self.q_net.load_state_dict(params["q_net_state_dict"])
        self.target_q_net.load_state_dict(params["q_net_state_dict"])
        self.q_optim.load_state_dict(params["q_net_optimizer_state_dict"])
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def __sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    @override
    def update(self):
        critic_losses = []
        for g in range(self.gradient_updates):
            if self.per:
                (
                    b_obs,
                    b_actions,
                    b_rewards,
                    b_next_obs,
                    b_dones,
                    b_inds,
                ) = self.__sample_batch_experiences()
            else:
                (
                    b_obs,
                    b_actions,
                    b_rewards,
                    b_next_obs,
                    b_dones,
                ) = self.__sample_batch_experiences()

            sampled_w = (
                th.tensor(random_weights(dim=self.reward_dim, n=self.num_sample_w, dist="gaussian", rng=self.np_random))
                .float()
                .to(self.device)
            )  # sample num_sample_w random weights
            w = sampled_w.repeat_interleave(b_obs.size(0), 0)  # repeat the weights for each sample
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = (
                b_obs.repeat(self.num_sample_w, *(1 for _ in range(b_obs.dim() - 1))),
                b_actions.repeat(self.num_sample_w, 1),
                b_rewards.repeat(self.num_sample_w, 1),
                b_next_obs.repeat(self.num_sample_w, *(1 for _ in range(b_next_obs.dim() - 1))),
                b_dones.repeat(self.num_sample_w, 1),
            )

            with th.no_grad():
                target = self.ddqn_target(b_next_obs, w)
                    
                target_q = b_rewards + (1 - b_dones) * self.gamma * target

            q_values = self.q_net(b_obs, w)
            q_value = q_values.gather(
                1,
                b_actions.long().reshape(-1, 1, 1).expand(q_values.size(0), 1, q_values.size(2)),
            )
            q_value = q_value.reshape(-1, self.reward_dim)

            critic_loss = F.mse_loss(q_value, target_q)
            
            if self.homotopy_lambda > 0:
                wQ = th.einsum("br,br->b", q_value, w)
                wTQ = th.einsum("br,br->b", target_q, w)
                auxiliary_loss = F.mse_loss(wQ, wTQ)
                critic_loss = (1 - self.homotopy_lambda) * critic_loss + self.homotopy_lambda * auxiliary_loss

            self.q_optim.zero_grad()
            critic_loss.backward()
            if self.log and self.global_step % 100 == 0:
                wandb.log(
                    {
                        "losses/grad_norm": get_grad_norm(self.q_net.parameters()).item(),
                        "global_step": self.global_step,
                    },
                )
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.q_optim.step()
            critic_losses.append(critic_loss.item())

            if self.per:
                td_err = (q_value[: len(b_inds)] - target_q[: len(b_inds)]).detach()
                priority = th.einsum("sr,sr->s", td_err, w[: len(b_inds)]).abs()
                priority = priority.cpu().numpy().flatten()
                priority = (priority + self.replay_buffer.min_priority) ** self.per_alpha
                self.replay_buffer.update_priorities(b_inds, priority)

        if self.tau != 1 or self.global_step % self.target_net_update_freq == 0:
            polyak_update(self.q_net.parameters(), self.target_q_net.parameters(), self.tau)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_epsilon,
            )

        if self.homotopy_decay_steps is not None:
            self.homotopy_lambda = linearly_decaying_value(
                self.initial_homotopy_lambda,
                self.homotopy_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_homotopy_lambda,
            )

        if self.log and self.global_step % 100 == 0:
            wandb.log(
                {
                    "losses/critic_loss": np.mean(critic_losses),
                    "metrics/epsilon": self.epsilon,
                    "metrics/homotopy_lambda": self.homotopy_lambda,
                    "global_step": self.global_step,
                },
            )
            if self.per:
                wandb.log({"metrics/mean_priority": np.mean(priority)})
  
    def get_coefs(self,w,q_values):
        w_ada = self.w_net(w)
        scalarized_q_values = th.einsum("br,bar->ba", w, q_values)
        max_acts = th.argmax(scalarized_q_values, dim=1)
        # Action evaluated with the target network
        q_value_max_act = q_values.gather(1, max_acts.long().reshape(-1, 1, 1).expand(q_values.size(0), 1, q_values.size(2)))
        q_value_max_act = q_value_max_act.reshape(-1, self.reward_dim)
        
        # Get similarity gradient
        bili =  q_value_max_act/w
        g_s = bili.max(dim=-1).values.view(-1,1)*w-q_value_max_act
        # Min-norm objective (6)
        return self.lam*g_s+w_ada
    
    
    def solve_min_norm(self, g_s, G_i, lam):
        G = G_i+lam*g_s.unsqueeze(-1)
        coef, d = batched_min_norm_solver(G)
        return d
        
    
    
    def compute_actor_loss(self, act_prob, w, lam, q_values):   
        
        # value
        with th.no_grad():
            v_values = th.einsum("ba,bar->br", act_prob, q_values)
            # g_sim
            bili =  v_values/w
            g_s_coef = bili.max(dim=-1).values.view(-1,1)*w-v_values
            g_s_coef_n = g_s_coef/g_s_coef.norm(1,dim=-1,keepdim=True)
            arcs = th.arccos(F.cosine_similarity(v_values.detach(),w))/3.1416  # ang / pi
            ratio_LS = th.exp(-self.exp*arcs).unsqueeze(-1)
            sim_coef = ratio_LS*w + (1-ratio_LS)*g_s_coef_n
            # grad 
            q_val_norm = q_values-q_values.mean(dim=1,keepdim=True)
            g_sim = th.einsum("br,bar->ba", sim_coef, q_val_norm)
        
        d = self.solve_min_norm(g_sim, q_val_norm, lam)
        loss = -(act_prob*d).sum(-1).mean()
        
        return loss
        
        
        
        
    
    
    
    
    @override
    def update_preco(self):

        critic_losses = []
        for g in range(self.gradient_updates):
            if self.per:
                (
                    b_obs,
                    b_actions,
                    b_rewards,
                    b_next_obs,
                    b_dones,
                    b_inds,
                ) = self.__sample_batch_experiences()
            else:
                (
                    b_obs,
                    b_actions,
                    b_rewards,
                    b_next_obs,
                    b_dones,
                ) = self.__sample_batch_experiences()

            sampled_w = (
                th.tensor(random_weights(dim=self.reward_dim, n=self.num_sample_w, dist="gaussian", rng=self.np_random))
                .float()
                .to(self.device)
            )  # sample num_sample_w random weights
            w = sampled_w.repeat_interleave(b_obs.size(0), 0)  # repeat the weights for each sample
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = (
                b_obs.repeat(self.num_sample_w, *(1 for _ in range(b_obs.dim() - 1))),
                b_actions.repeat(self.num_sample_w, 1),
                b_rewards.repeat(self.num_sample_w, 1),
                b_next_obs.repeat(self.num_sample_w, *(1 for _ in range(b_next_obs.dim() - 1))),
                b_dones.repeat(self.num_sample_w, 1),
            )

            with th.no_grad():
                
                target = self.Preco_Target(b_next_obs, w)
                    
                target_q = b_rewards + (1 - b_dones) * self.gamma * target
      
            q_values = self.q_net(b_obs, w)
            
            
            q_value = q_values.gather(
                1,
                b_actions.long().reshape(-1, 1, 1).expand(q_values.size(0), 1, q_values.size(2)),
            )
            q_value = q_value.reshape(-1, self.reward_dim)
            
            critic_loss = F.mse_loss(q_value, target_q)
            
            '''with th.no_grad():
                w_coef = self.get_coefs(w, q_values)
            if self.homotopy_lambda > 0:
                wQ = th.einsum("br,br->b", q_value, w_coef)
                wTQ = th.einsum("br,br->b", target_q, w_coef)
                auxiliary_loss = F.mse_loss(wQ, wTQ)
                critic_loss = (1 - self.homotopy_lambda) * critic_loss + self.homotopy_lambda * auxiliary_loss'''

            self.q_optim.zero_grad()
            critic_loss.backward()
            if self.log and self.global_step % 100 == 0:
                wandb.log(
                    {
                        "losses/grad_norm": get_grad_norm(self.q_net.parameters()).item(),
                        "global_step": self.global_step,
                    },
                )
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.q_optim.step()
            
            # update actor
            a_logits = self.a_net(b_obs, w)
            probs = F.softmax(a_logits,dim=1)
            self.a_optim.zero_grad()
            actor_loss = self.compute_actor_loss(probs, w, self.lam, q_values.detach())
            actor_loss.backward()
            self.a_optim.step()
            
            
            critic_losses.append(critic_loss.item())

            if self.per:
                td_err = (q_value[: len(b_inds)] - target_q[: len(b_inds)]).detach()
                priority = th.einsum("sr,sr->s", td_err, w[: len(b_inds)]).abs()
                priority = priority.cpu().numpy().flatten()
                priority = (priority + self.replay_buffer.min_priority) ** self.per_alpha
                self.replay_buffer.update_priorities(b_inds, priority)

        if self.tau != 1 or self.global_step % self.target_net_update_freq == 0:
            polyak_update(self.q_net.parameters(), self.target_q_net.parameters(), self.tau)
            polyak_update(self.a_net.parameters(), self.target_a_net.parameters(), self.tau)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_epsilon,
            )

        if self.homotopy_decay_steps is not None:
            self.homotopy_lambda = linearly_decaying_value(
                self.initial_homotopy_lambda,
                self.homotopy_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_homotopy_lambda,
            )

        if self.log and self.global_step % 100 == 0:
            wandb.log(
                {
                    "losses/critic_loss": np.mean(critic_losses),
                    "metrics/epsilon": self.epsilon,
                    "metrics/homotopy_lambda": self.homotopy_lambda,
                    "global_step": self.global_step,
                },
            )
            if self.per:
                wandb.log({"metrics/mean_priority": np.mean(priority)})
    
    
    @override
    def eval(self, obs: np.ndarray, w: np.ndarray) -> int:
        obs = th.as_tensor(obs).float().to(self.device)
        w = th.as_tensor(w).float().to(self.device)
        return self.max_action(obs, w)

    def act(self, obs: th.Tensor, w: th.Tensor) -> int:
        """Epsilon-greedily select an action given an observation and weight.

        Args:
            obs: observation
            w: weight vector

        Returns: an integer representing the action to take.
        """
        a_logits = self.a_net(obs, w)
        noise = -th.empty_like(a_logits).exponential_().log()  # Gumbel(0,1)
        ac = (a_logits + noise).argmax(dim=-1)
        return ac.detach().item()

    @th.no_grad()
    def max_action(self, obs: th.Tensor, w: th.Tensor) -> int:
        """Select the action with the highest Q-value given an observation and weight.

        Args:
            obs: observation
            w: weight vector

        Returns: the action with the highest Q-value.
        """
        a_logits = self.a_net(obs, w)
        max_act = a_logits.argmax()
        return max_act.detach().item()

    @th.no_grad()
    def envelope_target(self, obs: th.Tensor, w: th.Tensor, sampled_w: th.Tensor) -> th.Tensor:
        """Computes the envelope target for the given observation and weight.

        Args:
            obs: current observation.
            w: current weight vector.
            sampled_w: set of sampled weight vectors (>1!).

        Returns: the envelope target.
        """
        # Repeat the weights for each sample
        W = sampled_w.repeat(obs.size(0), 1)
        # Repeat the observations for each sampled weight
        next_obs = obs.repeat_interleave(sampled_w.size(0), 0)
        
        # Batch size X Num sampled weights X Num actions X Num objectives
        next_q_values = self.q_net(next_obs, W).view(obs.size(0), sampled_w.size(0), self.action_dim, self.reward_dim)
      
        # Scalarized Q values for each sampled weight
        scalarized_next_q_values = th.einsum("br,bwar->bwa", w, next_q_values)
        # Max Q values for each sampled weight
        max_q, ac = th.max(scalarized_next_q_values, dim=2)
        # Max weights in the envelope
        pref = th.argmax(max_q, dim=1)

        # MO Q-values evaluated on the target network
        next_q_values_target = self.target_q_net(next_obs, W).view(
            obs.size(0), sampled_w.size(0), self.action_dim, self.reward_dim
        )
        # Index the Q-values for the max actions
        max_next_q = next_q_values_target.gather(
            2,
            ac.unsqueeze(2).unsqueeze(3).expand(next_q_values.size(0), next_q_values.size(1), 1, next_q_values.size(3)),
        ).squeeze(2)
        # Index the Q-values for the max sampled weights
        max_next_q = max_next_q.gather(1, pref.reshape(-1, 1, 1).expand(max_next_q.size(0), 1, max_next_q.size(2))).squeeze(1)
        return max_next_q

    @th.no_grad()
    def ddqn_target(self, obs: th.Tensor, w: th.Tensor) -> th.Tensor:
        """Double DQN target for the given observation and weight.

        Args:
            obs: observation
            w: weight vector.

        Returns: the DQN target.
        """
        # Max action for each state
        q_values = self.q_net(obs, w)
      
        scalarized_q_values = th.einsum("br,bar->ba", w, q_values)
        max_acts = th.argmax(scalarized_q_values, dim=1)
        # Action evaluated with the target network
        q_values_target = self.target_q_net(obs, w)
        q_values_target = q_values_target.gather(
            1,
            max_acts.long().reshape(-1, 1, 1).expand(q_values_target.size(0), 1, q_values_target.size(2)),
        )
        q_values_target = q_values_target.reshape(-1, self.reward_dim)
        return q_values_target
    
    @th.no_grad()
    def Preco_Target(self, obs: th.Tensor, w: th.Tensor) -> th.Tensor:
        """Double DQN target for the given observation and weight.

        Args:
            obs: observation
            w: weight vector.

        Returns: the DQN target.
        """
        q_values_target = self.target_q_net(obs, w)
        a_logits = self.target_a_net(obs, w)
        probs = F.softmax(a_logits,dim=1)
      
        v_values_target = th.einsum("ba,bar->br", probs, q_values_target)
   
        return v_values_target
        
    @th.no_grad()
    def preco_target(self, obs: th.Tensor, w: th.Tensor) -> th.Tensor:
        """Double DQN target for the given observation and weight.

        Args:
            obs: observation
            w: weight vector.

        Returns: the DQN target.
        """
        q_values = self.q_net(obs, w)
        w_coef = self.get_coefs(w, q_values.detach())
        scalarized_q_values = th.einsum("br,bar->ba", w_coef, q_values)
        max_acts = th.argmax(scalarized_q_values, dim=1)
        # Action evaluated with the target network
        q_values_target = self.target_q_net(obs, w)
        q_values_target = q_values_target.gather(
            1,
            max_acts.long().reshape(-1, 1, 1).expand(q_values_target.size(0), 1, q_values_target.size(2)),
        )
        q_values_target = q_values_target.reshape(-1, self.reward_dim)
        return q_values_target
    
   
    
    
    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        ref_point: Optional[np.ndarray] = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        weight: Optional[np.ndarray] = None,
        total_episodes: Optional[int] = None,
        reset_num_timesteps: bool = True,
        eval_freq: int = 10000,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        reset_learning_starts: bool = False,
        verbose: bool = False,
    ):
        """Train the agent.

        Args:
            total_timesteps: total number of timesteps to train for.
            eval_env: environment to use for evaluation. If None, it is ignored.
            ref_point: reference point for the hypervolume computation.
            known_pareto_front: known pareto front for the hypervolume computation.
            weight: weight vector. If None, it is randomly sampled every episode (as done in the paper).
            total_episodes: total number of episodes to train for. If None, it is ignored.
            reset_num_timesteps: whether to reset the number of timesteps. Useful when training multiple times.
            eval_freq: policy evaluation frequency (in number of steps).
            num_eval_weights_for_front: number of weights to sample for creating the pareto front when evaluating.
            num_eval_episodes_for_front: number of episodes to run when evaluating the policy.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            reset_learning_starts: whether to reset the learning starts. Useful when training multiple times.
            verbose: whether to print the episode info.
        """
        if eval_env is not None:
            assert ref_point is not None, "Reference point must be provided for the hypervolume computation."
        
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist() if ref_point is not None else None,
                    "known_front": known_pareto_front,
                    "weight": weight.tolist() if weight is not None else None,
                    "total_episodes": total_episodes,
                    "reset_num_timesteps": reset_num_timesteps,
                    "eval_freq": eval_freq,
                    "num_eval_weights_for_front": num_eval_weights_for_front,
                    "num_eval_episodes_for_front": num_eval_episodes_for_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "reset_learning_starts": reset_learning_starts,
                }
            )

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        if reset_learning_starts:  # Resets epsilon-greedy exploration
            self.learning_starts = self.global_step
        self.env._cached_spec.max_episode_steps = 250
        eval_env = copy.deepcopy(self.env)

      
        num_episodes = 0
        
        eval_weights = w_test4()#w_test6()
        obs, _ = self.env.reset()

        w = weight if weight is not None else random_weights(self.reward_dim, 1, dist="gaussian", rng=self.np_random)
        tensor_w = th.tensor(w).float().to(self.device)
        lists = []
        for t_ in range(1, total_timesteps + 1):
            if total_episodes is not None and num_episodes == total_episodes:
                break

            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.act(th.as_tensor(obs).float().to(self.device), tensor_w)
           
            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)
            terminated = False
            truncated = False
            if t_%250==0:
                truncated = True
            self.global_step += 1

            self.replay_buffer.add(obs, action, vec_reward, next_obs, terminated)
            if t_%1000==0: 
                self.Qmem.append(self.q_net)
            if self.global_step >= self.learning_starts:
                self.update_preco()
            if t_%5000==0:
                if self.lam<=25:
                    self.lam += 0.02
                if self.exp<=15:
                    self.exp += 0.2
                print(t_,self.lam,self.exp,self.seed,"seed")
            if eval_env is not None and self.global_step % 30000 == 0 and t_>=150000:
                current_front = [
                    self.policy_eval(eval_env, weights=ew, num_episodes=num_eval_episodes_for_front, log=self.log)[2]
                    for ew in eval_weights
                ]
                
                filtered_front = list(filter_pareto_dominated(current_front))
                print(filtered_front)
                hv = hypervolume(ref_point, filtered_front)
                hn = np.array(current_front)/np.linalg.norm(np.array(current_front),axis=1).reshape(-1,1)
                Hr = np.diag(np.matmul(hn,np.array(eval_weights).T))
                hr = Hr.mean()
                cs = np.diag(cosine_similarity(np.array(current_front), np.array(eval_weights))).mean()
                print("HV:",hv,"CS:",cs,"Hr",hr,"step:",t_)
                lists.append((hv,cs,hr))
                print("results:", lists)
                

            if terminated or truncated:
   
                obs, _ = self.env.reset()
                num_episodes += 1
                self.num_episodes += 1

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], np.dot, w, self.global_step, verbose=verbose)

                if weight is None:
                    w = random_weights(self.reward_dim, 1, dist="gaussian", rng=self.np_random)
                    tensor_w = th.tensor(w).float().to(self.device)

            else:
                obs = next_obs
