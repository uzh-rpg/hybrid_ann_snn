from results.iccv.tools import *
from results.iccv.core import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

red_hex = '#D20000'
purple_hex = '#82218B'
snn_blue_hex = '#007aff'
cnn_blue_hex = '#03468F'
green_hex = '#007355'
snn_state_init_hex = '#f65353'
orange_hex = '#f59a23'

plt.rcParams.update({'font.family':'serif',
                     'font.size': 20})

cnn_100hz_camview3 = [5.03332938, 5.02943901, 5.03870562, 5.04159467, 5.03413161, 5.03561627,
 5.03155145, 5.03109182, 5.03819372, 5.03238965]
cnn_10_hz_camview3 = [5.07691781, 5.18712101, 5.33434984, 5.50652973, 5.69624012, 5.89592509,
 6.10114476, 6.31012702, 6.5220605, 6.7335798 ]
snn_pred_from_scratch = [45.61022856, 28.31567673, 21.26944424, 18.43567105, 17.21961705, 18.40257592,
 16.54691483, 16.08282803, 16.40341593, 15.48913346]
snn_delta_pred_w_init = [5.15675148, 5.19136022, 5.25637062, 5.28017713, 5.22805049, 5.19111861,
 5.15262202, 5.13566677, 5.13004327, 5.16797783]
snn_delta_pred_no_init = [5.09639431, 5.31262932, 5.36762717, 5.47672091, 5.56620267, 5.65820695,
 5.88289619, 6.21496164, 6.69374447, 7.20223486]
snn_scratch_state_init = [5.7328514, 5.60158339, 5.40656344, 5.36210251, 5.37948052, 5.39731583, 5.38588711, 5.4158566,  5.4193007,  5.54858841]
# snn_scratch_state_init = [5.6622742,  5.76188281, 5.62506444, 5.56305018, 5.53598944, 5.47479571, 5.39801166, 5.32154496, 5.26844267, 5.29509248]

root_dir = '/home/asude/thesis_final/master_thesis/results/iccv'

# fig1, ax1 = plt.subplots(figsize=(9, 6))
# ax1.plot(snn_pred_from_scratch, color=purple_hex, marker = '^', markersize = 10, linewidth=2,)
# ax1.plot(snn_delta_pred_w_init, color=orange_hex,  marker='^', markersize = 10, linewidth=3,)
# ax1.set_yscale('log')
# ax1.legend(['SNN (A)', 'Hybrid (D)'])
# labels_num = np.arange(1, 11, 1)
# labels = [str(label) for label in labels_num]
# ax1.set_xticks(ticks=np.arange(0, 10, 1), minor=False)
# ax1.set_xticklabels(labels=np.arange(10, 110, 10))
# ax1.set_xlabel('Time [ms]')
# ax1.set_ylabel('MPJPE (2D)')
# ax1.set_yticks(ticks=[5, 10, 20, 30, 40, 50])
# ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# plt.tight_layout()
# plt.savefig(root_dir + '/scratch_vs_hybrid.png', dpi=400)
# # # pdb.set_trace()


# CNN HYBRID COMPARISON
# plt.figure(figsize=(9, 6))
# plt.plot(cnn_10_hz_camview3, color=cnn_blue_hex, marker='s', markersize = 10, linewidth=2,)
# plt.plot(cnn_100hz_camview3, color=cnn_blue_hex,  marker='^', markersize = 10, linewidth=2,)
# plt.plot(snn_delta_pred_w_init, color=snn_state_init_hex,  marker='^', linestyle='--', markersize = 10, linewidth=2,)
# # plt.yscale('log')
# plt.legend(['CNN 10 Hz', 'CNN 100 Hz', 'Delta + State Init (Ours)'])
# plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1), fontfamily='serif')
# plt.xlabel('# of Time Steps [T=10 ms]')
# plt.ylabel('MPJPE (2D)')
# # plt.yticks(ticks=[5, 6, 7, 7.5])
# plt.tight_layout()
# plt.savefig(root_dir + '/cnn_hyrid_comp.png')
# pdb.set_trace()


# # ABLATION FIGURE
# fig1, ax1 = plt.subplots(figsize=(9, 6))
# ax1.plot(snn_scratch_state_init, color=red_hex, marker='^', markersize = 10, linewidth=2,)
# ax1.plot(snn_delta_pred_no_init, color=green_hex,  marker='^', markersize = 10, linewidth=2,)
# ax1.plot(snn_delta_pred_w_init, color=orange_hex,  marker='^',  markersize = 10, linewidth=3,)
# ax1.set_yscale('log')
# ax1.legend(['Hybrid - Output Init (B)', 'Hybrid - State Init (C)', 'Hybrid (D)'])
# ax1.set_xticks(ticks=np.arange(0, 10, 1), minor=False)
# ax1.set_xticklabels(labels=np.arange(10, 110, 10))
# ax1.set_xlabel('Time [ms]')
# ax1.set_ylabel('MPJPE (2D)')
# ax1.set_yticks(ticks=[5, 6, 7, 7.5])
# ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# plt.tight_layout()
# plt.savefig(root_dir + '/hybrid_ablation_plot.png', dpi=400)
# pdb.set_trace()


plt.figure(figsize=(9,6))
# plt.plot(snn_camview2_spikes_sorted.sum(axis=0)/num_layers, color=green_hex, marker = 'o')
plt.plot(snn_camview3_spikes_sorted.sum(axis=0)/num_layers, color=orange_hex, marker = '^', linewidth=2, markersize = 10)
# plt.plot(hybrid_camview2_spikes_sorted.sum(axis=0)/num_layers, color=green_hex,  marker='^', linestyle='--')
plt.plot(hybrid_camview3_spikes_sorted.sum(axis=0)/num_layers, color=purple_hex,  marker='^', linewidth=2, markersize = 10)

plt.legend(['SNN', 'Hybrid'])
plt.xticks(np.arange(0, 10, 1), np.arange(10, 110, 10))
plt.ylabel('Spike Firing Rate')
plt.xlabel('Time [ms]')
plt.tight_layout()
plt.savefig(root_dir + '/spike_firing_rate_per_timestep.pdf')

pdb.set_trace()
# plt.figure()
# plt.plot(snn_camview2_spikes_sorted.sum(axis=1)/num_bins, color=green_hex, marker = 'o')
# plt.plot(snn_camview3_spikes_sorted.sum(axis=1)/num_bins, color=orange_hex, marker = 'o')
# plt.plot(hybrid_camview2_spikes_sorted.sum(axis=1)/num_bins, color=green_hex,  marker='^', linestyle='--')
# plt.plot(hybrid_camview3_spikes_sorted.sum(axis=1)/num_bins, color=orange_hex,  marker='^', linestyle='--')

# plt.legend(['Cam 2 before', 'Cam 3 before','Cam 2 after','Cam 3 after'])
# plt.xticks(np.arange(0, 9, 1), np.arange(1, 10, 1))
# plt.ylabel('Spike Firing Rate Per Layer')
# plt.xlabel('Num layer')
# plt.tight_layout()
# plt.savefig(root_dir + '/spike_firing_rate_per_layer.png')
