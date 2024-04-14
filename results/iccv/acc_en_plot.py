import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import pdb 

import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

# plt.style.use(['science', 'grid'])
# mpl.rc('text', usetex=True)
# plt.style.use(['science','ieee'])
plt.style.use(['science','no-latex'])


snn_blue_hex = '#007aff'
cnn_blue_hex = '#03468F'
green_hex = '#007355'
snn_state_init_hex = '#f65353'
orange_hex = '#f59a23'

# cmap = mpl.colormaps['turbo']
colors = [orange_hex, orange_hex, cnn_blue_hex, cnn_blue_hex, green_hex, green_hex, snn_state_init_hex, snn_state_init_hex]
#cmap = mpl.colormaps['jet']

# DHP19
our_power_100hz = 0.424
our_power_200hz = 0.432
ann_power_100hz = 3.718
ann_power_200hz = 3.718*2

dhp19_power_100hz = 0.431
dhp19_power_200hz = 0.431*2
baldwin_power_100hz = 1.605
baldwin_power_200hz = 1.605*2

our_acc_100hz = 5.19
our_acc_200hz = 5.16
dhp19_acc = 7.72 
ann_acc = 5.03
baldwin_acc = 5.98


x_power = np.array([our_power_100hz, our_power_200hz, ann_power_100hz, ann_power_200hz, dhp19_power_100hz, dhp19_power_200hz, baldwin_power_100hz, baldwin_power_200hz])
y_acc = np.array([our_acc_100hz, our_acc_200hz, ann_acc, ann_acc, dhp19_acc, dhp19_acc, baldwin_acc, baldwin_acc])
mdl_size = np.array([4, 12 ,4, 12, 4, 12, 4, 12, 4, 12])
area = mdl_size*18

classes = ['Hybrid (Ours)', 
'_nolegend_',
'ANN (Ours)', 
'_nolegend_', 
'Calabrese', 
'_nolegend_', 
'Baldwin', 
'_nolegend_']



# for idx, color in enumerate(colors):
#     plt.scatter(x=x_power[idx], y=y_acc[idx], s=area[idx], c=color, label=classes[idx], alpha=0.5)
# plt.legend(loc=(1,0.6))
# plt.xlim([0, 8])
# plt.xlabel('Power (W)')
# plt.ylabel('MPJPE (2D)')
# plt.grid('on')
# plt.legend()
# plt.gca().add_artist(legend1)

# # plt.legend(['100 Hz', '200 Hz'], )
root_dir = '/home/asude/master_thesis/results/iccv/'
# plt.savefig(root_dir + 'acc.pdf')


colors_black = ['black', 'black']
x_power = np.array([0,0])
y_acc = np.array([0,0])
mdl_size = np.array([4, 12])
area = mdl_size*18
classes = ['100 Hz', '200 Hz']
# pdb.set_trace()
for idx, color in enumerate(colors_black): 
    plt.scatter(x=x_power[idx], y=y_acc[idx], s=area[idx], c=colors_black[idx], label=classes[idx], alpha=0.5)

leg = plt.legend(loc=(1.0,0.04))
plt.savefig(root_dir + 'legend.png')

x_power = np.array([our_power_100hz, our_power_200hz, ann_power_100hz, ann_power_200hz, dhp19_power_100hz, dhp19_power_200hz, baldwin_power_100hz, baldwin_power_200hz])
y_acc = np.array([our_acc_100hz, our_acc_200hz, ann_acc, ann_acc, dhp19_acc, dhp19_acc, baldwin_acc, baldwin_acc])
mdl_size = np.array([4, 12 ,4, 12, 4, 12, 4, 12, 4, 12])
area = mdl_size*18

classes = [
'Hybrid (Ours)', 
'_nolegend_',
'ANN (Ours)', 
'_nolegend_', 
'Calabrese', 
'_nolegend_', 
'Baldwin', 
'_nolegend_',
'_nolegend_'
]
# '_nolegend_']

# plt.figure()
for idx, color in enumerate(colors):
    plt.scatter(x=x_power[idx], y=y_acc[idx], s=area[idx], c=color, label=classes[idx], alpha=0.5)
plt.legend(loc=(0.46,0.37))
plt.xlim([0, 8])
plt.ylim([4.80,8])
plt.xlabel('Power (W)')
plt.ylabel('MPJPE (2D)')
plt.grid('on')
# leg1 = plt.legend(classes)
# plt.gca().add_artist(leg)
# plt.gca().add_artist(leg2)

# plt.legend(['100 Hz', '200 Hz'], )
root_dir = '/home/asude/master_thesis/results/iccv/'
plt.savefig(root_dir + 'acc.pdf')