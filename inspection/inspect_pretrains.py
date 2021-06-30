#%%

import glob
import os

#%% 

files = glob.glob('../pre_trained/*.pth')

with open('model_sizes.txt', 'w') as f:
    f.write('model_name, size\n')
    for file in files:
        line = os.path.split(file)[-1] + f", {os.stat(file).st_size / (1024 * 1024): .3f}" + ' MB\n'
        f.write(line)
        
# %%
