#%% Imports

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import os
import numpy as np

# %%

class InspectTrain():
    def __init__(self, data_identifier_source, results_dir="../results/", extension=".csv"):
        self.results_dir = results_dir
        self.source = data_identifier_source
        self.extension = extension
        self.train_dir = os.path.join(results_dir, data_identifier_source + '/train/')
        self.data, self.metrics = self.read_data()
        self.runs = list(self.data.keys())
        
    def read_data(self):
        
        dfs = dict()
        cols = list()
        
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(self.extension):
                    path2csv = os.path.join(root, f)
                    run_name = path2csv.split('/')[-2]
                    df = pd.read_csv(path2csv) # must be changed for other exts
                    dfs[str(run_name)] = df
                    cols = cols + list(df.columns)
                    
        cols = list(set(cols))
        
        for col in cols:
            if 'unnamed' in col.lower():
                cols.remove(col)
                for key in dfs.keys():
                    df = dfs[key]
                    if col in df.columns:
                        df.drop(col, inplace=True, axis=1)
                        dfs[key] = df
                    
        return dfs, cols
    
    def show(self, param, subparam=None, xlabel='epoch'):
        
        if subparam is not None:
            if not isinstance(subparam, list):
                subparam = [subparam]  
        
        plot_dict = dict()
        
        if param in self.metrics:
            if subparam is None:
                subparam = self.runs
            for run in subparam:
                df = self.data[run]
                if param in df.columns:
                    plot_dict[run] = df[param]
            
        elif param in self.runs:
            df = self.data[param]
            if subparam is None:
                subparam = df.columns
            for metric in subparam:
                plot_dict[metric] = df[metric]    
            
        else:
            print('No such run or metric name \'%s\' has been found!'.format(param))
            return -1
        
        fig, ax = plt.subplots(1)
            
        for key in plot_dict.keys():
            
            if key == 'save' or param =='save':
                y = np.random.rand(1)*plot_dict[key]
                x = np.array(list(range(len(y))))
                mask = y > 0
                ax.scatter(x[mask], y[mask], label=key)
            else:
                ax.plot(plot_dict[key], label=key)
            
        ax.set_xlabel(xlabel)
        ax.set_title(param)
        ax.legend()
            
        return plot_dict
    
    def __getitem__(self, idx):
        
        if isinstance(idx, int):
            key = self.data.keys()[idx]
        elif isinstance(idx, str):
            key = idx
        else:
            return -1
            
        return self.data[key]

# %%

data_identifier_source = 'abide_caltech'

it = InspectTrain(data_identifier_source)
# %%

it.show('save')
# %%
