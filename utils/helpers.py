from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np
import torch

def get_obj_grad_kl(ref_vec_KL, returns):
    returns = returns/(returns.norm(p=1,dim=-1).reshape(-1,1)+0.0001)
    grad_0 = ref_vec_KL[0]/returns[:,0]
    grad_1 = ref_vec_KL[1]/returns[:,1]
    return torch.cat([grad_0.reshape(-1,1),grad_1.reshape(-1,1)],dim=1)
    
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
