"""
TODO: MODIFY TO FILL IN YOUR BC IMPLEMENTATION
"""
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import rollout
from pdb import set_trace as debug
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=50, 
                       batch_size=32, shuffle_method='episodic'):
    # Shuffling method allows for two options:
    #   complete: shuffle the entire dataset
    #   episodic: shuffle the trajectory order, but keep the trajectory in order
    
    # Construct flattened expert_data (same method as main.py)
    flattened_expert = {'observations': [], 
                        'actions': []}
    for expert_path in expert_data:
        for k in flattened_expert.keys():
            flattened_expert[k].append(expert_path[k])
    for k in flattened_expert.keys():
        flattened_expert[k] = np.concatenate(flattened_expert[k])
    num_batches = len(flattened_expert['observations']) // batch_size
    episode_lengths = np.array([len(expert_path['observations']) for expert_path in expert_data]) # allows for dynamic episode lengths
    idxs = np.arange(len(flattened_expert['observations']))
    idxs_ep = np.arange(len(expert_data))
    
    # Epoch loop
    optimizer = optim.Adam(list(policy.parameters())) # removed lr argument since it is not specified in dagger.py, want apples-to-apples comparison
    losses = []
    for epoch in range(num_epochs): 
        
        # Set shuffle method
        if shuffle_method == 'complete':
            np.random.shuffle(idxs)
        elif shuffle_method == 'episodic':
            idxs = []
            np.random.shuffle(idxs_ep)
            for i in idxs_ep:
                base_idx = sum(episode_lengths[:i])
                idxs += list(range(base_idx, base_idx + episode_lengths[i]))
            idxs = np.array(idxs)
            
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            # Acquire new batch
            obs_batch = torch.from_numpy(flattened_expert['observations'][idxs[(i*batch_size) + np.arange(batch_size)],:]).float().to(device)
            act_batch = torch.from_numpy(flattened_expert['actions'][idxs[(i*batch_size) + np.arange(batch_size)],:]).float().to(device)   
            # Compute loss and backprop/optimize
            loss = -torch.sum(policy.log_prob(obs_batch,act_batch))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print('[%d] loss: %.8f' %
            (epoch, running_loss / 10.))
        losses.append(running_loss)
        
    return losses