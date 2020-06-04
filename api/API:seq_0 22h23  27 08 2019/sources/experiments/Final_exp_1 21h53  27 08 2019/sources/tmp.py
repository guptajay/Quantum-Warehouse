#%%
import numpy as np
batch_size = 32
nb_actions = 10
targets = np.zeros((batch_size, nb_actions))
dummy_targets = np.zeros((batch_size,))
masks = np.zeros((batch_size, nb_actions))
Rs = [np.random.randint(10) for i in range(batch_size) ]
action_batch = [np.random.randint(10) for i in range(batch_size) ]
for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
    target[action] = R
    dummy_targets[idx] = R
    mask[action] = 1.  # enable loss for this specific action
targets = np.array(targets).astype('float32')
masks = np.array(masks).astype('float32')

#%%
