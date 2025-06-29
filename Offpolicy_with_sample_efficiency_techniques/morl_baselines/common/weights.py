"""Utilities related to weight vectors."""

from functools import lru_cache
from typing import List, Optional

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions


def random_weights(
    dim: int, n: int = 1, dist: str = "dirichlet", seed: Optional[int] = None, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Generate random normalized weight vectors from a Gaussian or Dirichlet distribution alpha=1.

    Args:
        dim: size of the weight vector
        n : number of weight vectors to generate
        dist: distribution to use, either 'gaussian' or 'dirichlet'. Default is 'dirichlet' as it is equivalent to sampling uniformly from the weight simplex.
        seed: random seed
        rng: random number generator
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if dist == "gaussian":
        w = rng.standard_normal((n, dim))
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1, keepdims=True)
    elif dist == "dirichlet":
        w = rng.dirichlet(np.ones(dim), n)
    else:
        raise ValueError(f"Unknown distribution {dist}")

    if n == 1:
        return w[0]
    return w
    
def w_test4():
        test_res = 10
        r_dim=4
        ref_vec_list = []
        for i_ep in range(test_res+1):
            for j_ep in range(test_res+1-i_ep):
                for k_ep in range(test_res+1-i_ep-j_ep):
                            ref_vec = np.zeros(r_dim) 
                            ref_vec[0] = i_ep/test_res
                            ref_vec[1] = j_ep/test_res
                            ref_vec[2] = k_ep/test_res
                            ref_vec[-1]= 1-ref_vec[:-1].sum()
                            ref_vec_list.append(ref_vec)
        return ref_vec_list

def w_test6():
        test_res = 10
        r_dim=6
        ref_vec_list = []
        for i_ep in range(test_res+1):
            for j_ep in range(test_res+1-i_ep):
                for k_ep in range(test_res+1-i_ep-j_ep):
                    for l_ep in range(test_res+1-i_ep-j_ep-k_ep):
                        for m_ep in range(test_res+1-i_ep-j_ep-k_ep-l_ep):
                            ref_vec = np.zeros(r_dim) 
                            ref_vec[0] = i_ep/test_res
                            ref_vec[1] = j_ep/test_res
                            ref_vec[2] = k_ep/test_res
                            ref_vec[3] = l_ep/test_res
                            
                            ref_vec[4]= m_ep/test_res
                            ref_vec[-1]= 1-ref_vec[:-1].sum()
                            ref_vec_list.append(ref_vec)
        return ref_vec_list

@lru_cache
def equally_spaced_weights(dim: int, n: int, seed: int = 42) -> List[np.ndarray]:
    """Generate weight vectors that are equally spaced in the weight simplex.

    It uses the Riesz s-Energy method from pymoo: https://pymoo.org/misc/reference_directions.html

    Args:
        dim: size of the weight vector
        n: number of weight vectors to generate
        seed: random seed
    """
    return list(get_reference_directions("energy", dim, n, seed=seed))


def extrema_weights(dim: int) -> List[np.ndarray]:
    """Generate weight vectors in the extrema of the weight simplex. That is, one element is 1 and the rest are 0.

    Args:
        dim: size of the weight vector
    """
    return list(np.eye(dim, dtype=np.float32))
