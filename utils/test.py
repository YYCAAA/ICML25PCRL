import numpy as np

def test(cfg, env, agent):
    print("testing！")
    rewards = []  
    steps = []
    ref_ind = 0
    ref_vec_list = []
    mean_rs=[]
    for i_ep in range(cfg.test_res+1):
        for j_ep in range(cfg.test_res+1-i_ep):
           
            sum_r = np.zeros(cfg.r_dim)
            ep_reward = 0  
            ep_step = 0
            state = env.reset()[0]  
            ref_vec = np.zeros(cfg.r_dim) 
            ref_vec[0] = i_ep/cfg.test_res
            ref_vec[1] = j_ep/cfg.test_res
            ref_vec[-1]= 1-ref_vec[:-1].sum()
            ref_vec_list.append(ref_vec/np.linalg.norm(ref_vec))
            
            for _ in range(cfg.test_eps):
                state = env.reset()[0]  
                for _ in range(cfg.max_steps):
                    ep_step+=1
                    state = np.concatenate([state,ref_vec])
                    action = agent.predict_action(state) 
                    next_state, reward, done, _,_ = env.step(action)  
                    #reward = reward[1:]
                    state = next_state 
                    ep_reward += reward 
                    if done:
                        break
            

            steps.append(ep_step/cfg.test_eps)
            rewards.append(ep_reward/cfg.test_eps)
            print(i_ep+1,ep_reward[[0,1,2]]/cfg.test_eps,ref_vec)
            sum_r+=ep_reward[[0,1,2]]
            mean_rs.append(sum_r/cfg.test_eps)
        
    print("test done")
    env.close()
    return {'rewards':rewards}, mean_rs, ref_vec_list

def test4(cfg, env, agent):
    print("testing！")
    rewards = []  
    steps = []
    ref_ind = 0
    ref_vec_list = []
    mean_rs=[]
    for i_ep in range(cfg.test_res+1):
        print(i_ep+1)
        for j_ep in range(cfg.test_res+1-i_ep):
            for k_ep in range(cfg.test_res+1-i_ep-j_ep):
                sum_r = np.zeros(cfg.r_dim)
                ep_reward = 0  
                ep_step = 0
                state = env.reset()[0]  
                ref_vec = np.zeros(cfg.r_dim) 
                ref_vec[0] = i_ep/cfg.test_res
                ref_vec[1] = j_ep/cfg.test_res
                ref_vec[2] = k_ep/cfg.test_res
                ref_vec[-1]= 1-ref_vec[:-1].sum()
                ref_vec_list.append(ref_vec/np.linalg.norm(ref_vec))
                
                for _ in range(cfg.test_eps):
                    state = env.reset()[0] 
                    for _ in range(cfg.max_steps):
                        ep_step+=1
                        state = np.concatenate([state,ref_vec])
                        action = agent.predict_action(state) 
                        next_state, reward, done, _,_ = env.step(action) 
                        #reward = reward[1:]
                        state = next_state  
                        ep_reward += reward  
                        if done:
                            break
                

                steps.append(ep_step/cfg.test_eps)
                rewards.append(ep_reward/cfg.test_eps)
                #print(i_ep+1,ep_reward[[0,1,2,3]]/cfg.test_eps,ref_vec)
                sum_r+=ep_reward[[0,1,2,3]]
                mean_rs.append(sum_r/cfg.test_eps)
            
              
    print("test done!")
    env.close()
    return {'rewards':rewards}, mean_rs, ref_vec_list

def test5(cfg, env, agent):
    print("testing！")
    rewards = []  
    steps = []
    ref_ind = 0
    ref_vec_list = []
    mean_rs=[]
    for i_ep in range(cfg.test_res+1):
        for j_ep in range(cfg.test_res+1-i_ep):
            print(i_ep+1)
            for k_ep in range(cfg.test_res+1-i_ep-j_ep):
                for l_ep in range(cfg.test_res+1-i_ep-j_ep-k_ep):
                    sum_r = np.zeros(cfg.r_dim)
                    ep_reward = 0  
                    ep_step = 0
                    state = env.reset()[0]  
                    ref_vec = np.zeros(cfg.r_dim) 
                    ref_vec[0] = i_ep/cfg.test_res
                    ref_vec[1] = j_ep/cfg.test_res
                    ref_vec[2] = k_ep/cfg.test_res
                    ref_vec[3] = l_ep/cfg.test_res
                    ref_vec[-1]= 1-ref_vec[:-1].sum()
                    ref_vec_list.append(ref_vec/np.linalg.norm(ref_vec))
                    
                    for _ in range(cfg.test_eps):
                        state = env.reset()[0] 
                        for _ in range(cfg.max_steps):
                            ep_step+=1
                            state = np.concatenate([state,ref_vec])
                            action = agent.predict_action(state)  
                            next_state, reward, done, _,_ = env.step(action)  
                            #reward = reward[1:]
                            state = next_state 
                            ep_reward += reward  
                            if done:
                                break
                    

                    steps.append(ep_step/cfg.test_eps)
                    rewards.append(ep_reward/cfg.test_eps)
                    #print(i_ep+1,ep_reward[[0,1,2,3,4]]/cfg.test_eps,ref_vec)
                    sum_r+=ep_reward[[0,1,2,3,4]]
                    mean_rs.append(sum_r/cfg.test_eps)
                
    print("test done!")
    env.close()
    return {'rewards':rewards}, mean_rs, ref_vec_list

def test6(cfg, env, agent):
    print("testing！")
    rewards = []  
    steps = []
    ref_ind = 0
    ref_vec_list = []
    mean_rs=[]
    for i_ep in range(cfg.test_res+1):
        print(i_ep)
        for j_ep in range(cfg.test_res+1-i_ep):
            for k_ep in range(cfg.test_res+1-i_ep-j_ep):
                for l_ep in range(cfg.test_res+1-i_ep-j_ep-k_ep):
                    for m_ep in range(cfg.test_res+1-i_ep-j_ep-k_ep-l_ep):
                        sum_r = np.zeros(cfg.r_dim)
                        ep_reward = 0  
                        ep_step = 0
                        state = env.reset()[0]  
                        ref_vec = np.zeros(cfg.r_dim) 
                        ref_vec[0] = i_ep/cfg.test_res
                        ref_vec[1] = j_ep/cfg.test_res
                        ref_vec[2] = k_ep/cfg.test_res
                        ref_vec[3] = l_ep/cfg.test_res
                        ref_vec[-1]= 1-ref_vec[:-1].sum()
                        ref_vec_list.append(ref_vec/np.linalg.norm(ref_vec))
                        
                        for _ in range(cfg.test_eps):
                            state = env.reset()[0] 
                            for _ in range(cfg.max_steps):
                                ep_step+=1
                                state = np.concatenate([state,ref_vec])
                                action = agent.predict_action(state)  
                                next_state, reward, done, _,_ = env.step(action)  
                                #reward = reward[1:]
                                state = next_state  
                                ep_reward += reward 
                                if done:
                                    break
                        

                        steps.append(ep_step/cfg.test_eps)
                        rewards.append(ep_reward/cfg.test_eps)
                        #print(i_ep+1,ep_reward[[0,1,2,3,4,5]]/cfg.test_eps,ref_vec)
                        sum_r+=ep_reward[[0,1,2,3,4,5]]
                        mean_rs.append(sum_r/cfg.test_eps)
                    
                       
    print("test done!")
    env.close()
    return {'rewards':rewards}, mean_rs, ref_vec_list

