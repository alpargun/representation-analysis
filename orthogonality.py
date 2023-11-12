
#%%

import numpy as np

#%% Init successor states - Random input for testing. Replace with real embeddings

np.random.seed(0)

num_frames = 100

repr = np.random.standard_normal(size=(num_frames,2048)) # 4 encoders
repr

# %% Calculate orthogonality

sum_total = 0

for i in range(num_frames):
    for j in range(i+1,num_frames):
        
        sum_total += abs(np.dot(repr[i], repr[j])) / (np.linalg.norm(repr[i], ord=2) * np.linalg.norm(repr[j], ord=2))

orthogonality = 1 - 2 / (num_frames * (num_frames-1)) * sum_total
orthogonality

# %%
