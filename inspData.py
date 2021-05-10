#%% 
from data import data_loader
import matplotlib.pyplot as plt
import time
from PIL import Image
import numpy as np

_, test_loader, _ = data_loader.load_datasets('nci', 1)
# %%

for c, data in enumerate(test_loader):
    data = data[0]
    im = Image.fromarray(np.array(255*data[-1,0,:,:])).convert('RGB')
    im.save("data/nci/testImages/{}.png".format(c+1)) 
    time.sleep(0.2)
    print(c)
    
# %%

V = [20, 20, 20, 20, 19, 20, 20, 15, 20, 20]
np.sum(V)
# %%
