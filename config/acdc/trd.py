#%% 

import glob

files = sorted(glob.glob("test/*.py"))

print(files)
print(len(files))

#%%

cstr = "no_slices = [8, 8, 9, 9, 8, 8, 10, 10, 14, 14, 8, 8, 9, 9, 10, 10, 7, 7, 13, 13, 6, 6, 14, 14, 6, 6, 10, 10, 10, 10, 9, 9, 8, 8, 15, 15, 9, 9, 10, 10] \n"

instr, outstr = 'nci', 'acdc'

for f in files:
    
    if f.endswith("_1.py"):
        continue

    fin = open(f, "rt")
    fout = open(f.replace(instr, outstr), "wt")

    for line in fin:
        if line.startswith("no_slices"):
            fout.write(cstr)
        else:
            fout.write(line.replace(instr, outstr))
        
    fin.close()
    fout.close()
# %%

files2del = [file for file in glob.glob("test/*.py") if instr in file]

for f in files2del:
    os.remove(f)
# %%

