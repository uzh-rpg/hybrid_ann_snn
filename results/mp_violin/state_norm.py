import pickle 
import pdb 
import numpy as np
from scipy.stats import norm
import statistics
import scienceplots
import matplotlib.pyplot as plt
plt.style.use(['science','no-latex'])


bin_edges = np.arange(-20,20.01,0.01)
pickle_path = '/home/asude/master_thesis/results/mp_violin/file.pkl'
with open(pickle_path, 'rb') as handle: 
    hist_dict = pickle.load(handle)

cdf = {}
random_from_cdf = {}
values = np.random.rand(100000)
bin_midpoints = bin_edges[:-1] + np.diff(bin_edges)/2
for key in hist_dict.keys():
    cdf[key] = np.cumsum(hist_dict[key])
    cdf[key] = cdf[key] / cdf[key][-1]

    value_bins = np.searchsorted(cdf[key], values)
    random_from_cdf[key] = bin_midpoints[value_bins]

states = []
for key in random_from_cdf.keys(): states = np.concatenate((states, random_from_cdf[key]))

mean = statistics.mean(states)
sd = statistics.stdev(states)
plt.scatter(states, norm.pdf(states, mean, sd), s=2, color='indigo', marker='')
plt.axvline(1)
plt.xlim(-4, 4)
plt.savefig('norm.png')

# pdb.set_trace()
