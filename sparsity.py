#%%

import numpy as np

#%% Init successor states - Random input for testing. Replace with real embeddings

np.random.seed(0)

num_frames = 100

repr = np.random.standard_normal(size=(num_frames,2048)) # 4 encoders
repr

#%% Calculate sparsity

tolerance = -10 # the other paper uses 10

sparsity = (repr < 10**tolerance).sum() / (2048 * num_frames)
sparsity

# %%
