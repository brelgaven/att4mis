#%%

import glob

files = sorted(glob.glob("config/abide_caltech/test/*.py"))
cmds = ["sbatch ctun_test.sh " + file for file in files]

c = ' && '.join(cmds)
# %%

