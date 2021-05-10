#%% 
from importlib.machinery import SourceFileLoader
import copy

test_cfg_path = './config/abide_caltech/cfg_test_abide_caltech_try.py'
test_cfg_name = test_cfg_path.split('/')[-1].rstrip('.py')

test_config = SourceFileLoader(test_cfg_name, test_cfg_path).load_module()

dir(test_config)
print(test_config.__file__)
# %%

train_id = getattr(test_config, 'train_id', None)

cfg_tr_path = './results/' + test_config.data_identifier_source + '/train/' + train_id + '/cfg.py'

cfg_tr_name = cfg_tr_path.split('/')[-1].rstrip('.py')
train_config = SourceFileLoader(cfg_tr_name, cfg_tr_path).load_module()

dir(train_config)
# %%

all_attr = list(set(dir(test_config) + dir(train_config)))
print(all_attr)
# %%

def merge_modules(*modules):
    
    module_list = list(modules)
    
    if len(module_list) == 1:
        return module_list[0]
    
    module_root = module_list[0]
    module_add = module_list[1]

    for attr in list(set(dir(module_root) + dir(module_add))):
        if not hasattr(module_root, attr):
            attr_value = getattr(module_add, attr, None)
            setattr(module_root, attr, attr_value)
    
    module_list[1] = module_root        
    module_list.pop(0)
    
    return merge_modules(*module_list)

#%% 

merged = merge_modules(test_config, train_config, copy)
# %%

dir(merged)
# %%
print(merged.__name__)
# %%
