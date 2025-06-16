import argparse
from utils.config import Config
from utils.train import train
from utils.test import test, test4, test5, test6
testfs = {3: test, 4: test4, 5: test5, 6: test6}
from utils.plot import plot_rewards
from utils.env import env_agent_config
import numpy as np
import sys
from pygmo import hypervolume

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1)
    parser.add_argument("--r", default=6)
    parser.add_argument("--m", default='PreCo')
    args = parser.parse_args()

    cfg = Config()
    cfg.MO_algo_name = args.m
    cfg.seed = int(args.seed)
    cfg.r_dim = int(args.r)
    env, agent = env_agent_config(cfg)

  

    best_agent, res_dic, Hs = train(cfg, env, agent)
    res_dic, mean_rs, refs = testfs[cfg.r_dim](cfg, env, best_agent)

    ref_point = np.zeros(cfg.r_dim)
    hn = mean_rs / np.linalg.norm(mean_rs, axis=1).reshape(-1, 1)
    Hr = np.diag(np.matmul(hn, np.array(refs).T))
    hvfast = hypervolume(-np.array(mean_rs))
    v = hvfast.compute(ref_point)
    print("HV:", v, "HR:", Hr.sum())
    print(cfg.seed, "seed")

    plot_rewards(res_dic['rewards'], cfg, tag="train")
    np.set_printoptions(threshold=sys.maxsize)
    print(repr(np.array(mean_rs)), repr(np.array(refs)))

if __name__ == "__main__":
    main() 
