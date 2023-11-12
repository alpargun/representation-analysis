

#%%

import numpy as np
np.random.seed(0)

#%% Init successor states - Random input for testing. Replace with real embeddings

np.random.seed(0)

num_frames = 100

all_repr = np.random.rand(2, num_frames, 2048) # successor representations for 1 scenario
succ_repr = all_repr[0]
random_repr = all_repr[1]
random_repr

#%% Calculate the sum of distances for successor states

sum_total_succ = 0
prev_repr = succ_repr[0]

for repr in succ_repr[1:]:
    dist_repr = repr - prev_repr
    norm_repr = np.linalg.norm(dist_repr, ord=2)
    sum_total_succ += norm_repr

    prev_repr = repr

sum_total_succ

#%% Calculate the sum of distances for random states

sum_total_rand = 0

# Calculate the sum of distances for successor states
for repr_succ, repr_rand in zip(succ_repr[1:], random_repr[1:]):
    dist_repr = repr_succ - repr_rand
    norm_repr = np.linalg.norm(dist_repr, ord=2)
    sum_total_rand += norm_repr

sum_total_rand


#%% Calculate dynamics awareness

dynamics_awareness = (sum_total_rand - sum_total_succ) / sum_total_rand
dynamics_awareness


# %%