#%% 

import glob

files = sorted(glob.glob("test/*.py"))

print(files)
print(len(files))

#%%

cstr = "\n"
cstr2 = "image_size = (256, 256, 256) \n"

instr, outstr = 'acdc', 'abide_caltech'

for f in files:
    
    if f.endswith("_0.py"):
        continue

    fin = open(f, "rt")
    fout = open(f.replace(instr, outstr), "wt")

    for line in fin:
        if line.startswith("no_slices"):
            fout.write(cstr)
        elif line.startswith("image_size"):
            fout.write(cstr2)
        else:
            fout.write(line.replace(instr, outstr))
        
    fin.close()
    fout.close()
# %%

files2del = [file for file in glob.glob("test/*.py") if instr in file]

for f in files2del:
    os.remove(f)
# %%

