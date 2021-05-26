#%% Imports

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import os
from glob import glob
import numpy as np

# %%
N_COLORS = 30
viridis = matplotlib.cm.get_cmap('viridis', N_COLORS)
cls = np.array(viridis.colors[:,:3]*255).astype(int)
cs = ["#{:02x}{:02x}{:02x}".format(r,g,b) for r,g,b in cls]

#%%

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
        
        for root, _, files in os.walk(self.train_dir):
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
    
    def show(self, param, subparam=None, xlabel='epoch', show_stats=False):
        
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
            
        q = 0
        #cs = list(matplotlib.colors.TABLEAU_COLORS.values())
        for key in plot_dict.keys():
            
            if show_stats and (key != 'save'):
                min_val = np.min(plot_dict[key])
                max_val = np.max(plot_dict[key])
                std_val = np.std(plot_dict[key])
                avg_val = np.mean(plot_dict[key])
                med_val = np.median(plot_dict[key])
                print(key, ':', '{:.5f} | '.format(med_val),
                      '{:.5f} +/- {:.5f}'.format(avg_val, std_val),
                      '({:.5f}-{:.5f})'.format(min_val, max_val))
            
            if key == 'save' or param =='save':
                x = np.where(plot_dict[key] > 0)
                ax.vlines(x, 0.4, 0.6, label=key, alpha=0.9, colors=cs[q])
            else:
                ax.plot(plot_dict[key], label=key, alpha=0.6, c=cs[q])
            q += 1
            
        ax.set_xlabel(xlabel)
        ax.set_title((self.source + ': ' + param))
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
    
#%% 

class InspectTest():
    
    def __init__(self, data_identifier_source, results_dir="../results/", extension=".txt"):
        self.data_source = data_identifier_source
        self.test_dir = os.path.join(results_dir, data_identifier_source + '/test/')
        self.extension = extension
        self.scores = self.read_scores()
        self.summary = self.summarize()
        
    def read_scores(self):
        
        scores = dict()
        paths = glob(self.test_dir + '/*/*'+ self.extension)
        paths = sorted(paths, key=str.lower)
        
        for path in paths:
            run_name = path.split('/')[-2]
            df = pd.read_csv(path, sep="\t", header=None)
            scores_array = np.array(df[1])
            if list(df[0])[-1].lower() == 'average':
                scores_array = scores_array[:-1]
            
            scores[str(run_name)] = scores_array
        
        scores = pd.DataFrame.from_dict(scores)
        
        return scores

    def summarize(self):
        
        summary = self.scores.describe().transpose()

        return summary
    
    def show(self, inc=None):
        
        n_runs = len(self.summary)
        x_pos = np.array(list(range(n_runs)))
        
        runs = self.summary.index.values
        means = self.summary['mean']
        stds = self.summary['std']
        
        if inc is not None:
            
            filt = [inc in run for run in runs]
            runs = [i for (i, v) in zip(runs, filt) if v]
            means = [i for (i, v) in zip(means, filt) if v]
            stds = [i for (i, v) in zip(stds, filt) if v]
        
        fig, ax = plt.subplots()
        ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=5)
        ax.set_ylabel('Dice Scores')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(runs, rotation=70, ha='right')
        ax.set_title('Dice Scores for ' + self.data_source)
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.show()
        
        
#%% 

nci_test = InspectTest('nci')
nci_test.show()

#%% 

ac_test = InspectTest('abide_caltech')
ac_test.show()


# %%
