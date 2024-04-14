import pickle 
import pdb 
import numpy as np
import scienceplots

# from __future__ import division
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

colors = {'v_init_in':'mintcream','v_init_1':'springgreen', 'v_init_2':'mediumseagreen', 'v_init_3':'seagreen','v_init_4':'lime','v_init_5':'green','v_init_6':'darkgreen','v_init_7':'lightgreen','v_init_8':'forestgreen',}

plt.rcParams.update({'font.family':'serif',
                     'font.size': 25})

colors = ['#464196', '#3a2efe', '#2000b1', '#010fcc', '#464196', '#3a2efe', '#2000b1', '#010fcc', '#464196']
fig, ax = plt.subplots(figsize = (16,9))
plots = ax.violinplot([random_from_cdf[key] for key in random_from_cdf.keys()], vert = True, points=100, showextrema=False, showmeans=True)
for pc, color in zip(plots['bodies'], colors):
    pc.set_facecolor(color)

# Set the color of the median lines
plots['cmeans'].set_colors('black')

ax.set_ylim(-2.5,2.5)
ax.axhline(y = 1, linestyle='--', linewidth=2, color='indigo', label='Firing Threshold')
ax.set_ylabel('Membrane Potential (V)')
ax.set_xlabel('Layer #')
ax.set_xticks([1,2,3,4,5,6,7,8,9])
layers = [key for key in random_from_cdf.keys()]
# labels = ax.set_xticklabels(layers)
# for l in labels: l.update({"rotation": "vertical"})
plt.legend()
plt.tight_layout()
plt.savefig('MP_violin_scientific.pdf')

# plt.show()
# pdb.set_trace() (V)