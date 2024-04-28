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
                       batch_size=32):
    
    # Fill in your BC implementation in this function. 

    # Hint: Just flatten your expert dataset and use standard pytorch supervised learning code to train the policy. 
    optimizer = optim.Adam(list(policy.parameters()), lr=1e-4)
    idxs_ep = np.array(range(len(expert_data)))
    idxs_m = np.array(range(episode_length))
    max_samples = len(idxs_ep)*episode_length
    n_obs = expert_data[0]['observations'][0].shape[0]
    n_act = expert_data[0]['actions'][0].shape[0]
    losses = []
    for epoch in range(num_epochs): 
        np.random.shuffle(idxs_ep)
        np.random.shuffle(idxs_m)
        running_loss = 0.0
        idx_ep = 0 # keeps track of current episode
        idx_m  = 0 # keeps track of current sample within an episode
        epoch_reached = False
        for _ in range(max_samples):
            optimizer.zero_grad()
            # Obtain sequentially-sampled batch of expert data
            obs_batch = torch.zeros((batch_size, n_obs)).to(device)
            act_batch = torch.zeros((batch_size, n_act)).to(device)
            for j in range(batch_size):
                if idx_m > len(expert_data[idxs_ep[idx_ep]]['observations'])-1:
                    np.random.shuffle(idxs_m)
                    idx_ep += 1
                    idx_m = 0
                if idx_ep > len(expert_data)-1:
                    epoch_reached = True # flag to indicate epoch end (terminate batch early)
                    break
                obs_batch[j,:] = torch.from_numpy(expert_data[idxs_ep[idx_ep]]['observations'][idxs_m[idx_m]]).float()
                act_batch[j,:] = torch.from_numpy(expert_data[idxs_ep[idx_ep]]['actions'][idxs_m[idx_m]]).float()
                print(f'idx_ep: {idxs_ep[idx_ep]}, idx_m: {idxs_m[idx_m]}')
                idx_m += 1

            # Compute loss and backprop/optimize
            # loss = -policy.log_prob(obs_batch,act_batch).mean()
            loss = -policy.log_prob(obs_batch,act_batch).sum()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if epoch_reached:
                break
            
        # if epoch % 10 == 0:
        print('[%d] loss: %.8f' %
            (epoch, running_loss / 10.))
        losses.append(loss.item())
        
    # Plotting the losses against epoch iteration
    label = 'Reacher'
    sns.set_style("white")
    epochs = np.arange(0,num_epochs,1)
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(label+': Loss vs. Epoch')
    plt.savefig('figures/bc_loss.png')