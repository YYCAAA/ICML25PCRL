import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium import MORecordEpisodeStatistics

from morl_baselines.multi_policy.PCRL.PreCo import PreCo


def main():
    def make_env():
        env = mo_gym.make("mo-reacher-v4")
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        # env = mo_gym.LinearReward(env)
        return env

    env = make_env()
    eval_env = make_env()
    # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)
  
    agent = PreCo(
        env,
        max_grad_norm=0.1,
        learning_rate=3e-4,
        gamma=0.98,
        batch_size=64,
        net_arch=[256, 256, 256],
        buffer_size=int(2e6),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=50000,
        initial_homotopy_lambda=0.0,
        final_homotopy_lambda=1.0,
        homotopy_decay_steps=10000,
        learning_starts=100,
        envelope=True,
        gradient_updates=1,
        target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
        tau=1,
        log=True,
        project_name="MORL-Baselines",
        experiment_name="PreCo",
    )

    agent.train(
        total_timesteps=3000000,
        total_episodes=None,
        weight=None,
        eval_env=eval_env,
        ref_point=-100*np.ones(4),
        
        num_eval_weights_for_front=100,
        eval_freq=1000,
        reset_num_timesteps=False,
        reset_learning_starts=False,
    )


if __name__ == "__main__":
    main()
