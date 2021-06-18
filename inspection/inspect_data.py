#%% Imports

import pandas as pd
import numpy as np
import h5py

abide1_filename = '../data/abide/caltech/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_10.hdf5'
abide2_filename = '../data/abide/caltech/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_10_to_15.hdf5'
abide3_filename = '../data/abide/caltech/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_16_to_36.hdf5'
nci_filename = '../data/nci/data_2d_size_256_256_res_0.625_0.625_cv_fold_1.hdf5'
acdc_filename = '../data/acdc/data_2D_size_256_256_res_1.33_1.33_cv_fold_1.hdf5'

# %%

filename = acdc_filename

with h5py.File(filename, "r") as f:
    print("Keys: %s" % f.keys())
    im_vals = f['images_validation']
    print(im_vals.shape)
    for k in ['nx', 'ny', 'nz', 'px', 'py', 'pz']:
        for q in ['train', 'test', 'validation']:
            z = k + '_' + q
            p = np.array(f[z])
            print(z + ':[', p.min(), '-', p.max(), ']')
# %%

filename = acdc

with h5py.File(filename, "r") as f:
    print("Keys: %s" % f.keys())
    im_vals = f['images_validation']
    print(im_vals.shape)
    for q in ['train', 'test', 'validation']:
        for k in ['nx', 'ny', 'nz', 'px', 'py', 'pz']:
            z = k + '_' + q
            p = np.array(f[z])
            print(z + ':[', p.min(), '-', p.max(), ']')