class Config:
    def __init__(self) -> None:
        self.env_name = "fruit-tree-v0"
        self.r_dim = 6
        self.depth = 6
        self.probscale = 4
        self.new_step_api = False
        self.algo_name = "PPO"
        self.MO_algo_name = "PreCo"
        self.mode = "train"
        self.seed = 0
        self.device = "cuda"
        self.train_eps = 20
        self.ref_train_eps = 3000
        self.test_eps = 10
        self.test_res = 10
        self.max_steps = 100
        self.eval_eps = 5
        self.eval_per_episode = 10
        self.gamma = 0.99
        self.k_epochs = 3
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.eps_clip = 0.01
        self.entropy_coef = 0.001
        self.update_freq = 100
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256 

class Config_reacher:
    def __init__(self) -> None:
        self.env_name = "mo-reacher-v4" 
        self.r_dim = 4
        self.probscale = 4
        self.new_step_api = False 
        self.algo_name = "PPO"
        self.MO_algo_name = "PreCo"
        self.mode = "train" # train or test
        self.seed =0 
        self.device = "cuda" # device to use
        self.train_eps = 40 
        self.ref_train_eps = 800
        self.test_eps = 5
        self.test_res = 10
        self.max_steps = 250 
        self.eval_eps = 5 
        self.eval_per_episode = 40 

        self.gamma = 0.99 
        self.k_epochs = 2
        self.actor_lr = 0.0003 
        self.critic_lr = 0.0003 
        self.eps_clip = 0.01 # epsilon-clip
        self.entropy_coef = 0.001#0.001 
        self.update_freq = 250 
        self.actor_hidden_dim = 256 
        self.critic_hidden_dim = 256 