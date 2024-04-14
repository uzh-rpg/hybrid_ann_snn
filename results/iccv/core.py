import os 
from results.iccv.tools import *

num_bins = 10 
num_layers = 9

layer_names_snn = ['static_conv', 'down1', 'down2', 'down3', 'mid_residualBlock', 'residualBlock', 'up1', 'up2', 'up3']
layer_names = ['static_conv_snn', 'down1_snn', 'down2_snn', 'down3_snn', 'mid_residualBlock', 'residualBlock_snn', 'up1_snn', 'up2_snn', 'up3_snn']

root_dir = '/home/asude/master_thesis/experiments_dhp19'
snn_camview2_path = os.path.join(root_dir, 'snn_resblock_seperate_lif_camview2/results_snn_camview2.txt')
snn_camview3_path = os.path.join(root_dir, 'snn_resblock_seperate_lif_camview3/results_snn_camview3.txt')

hybrid_camview2_path= os.path.join(root_dir, 'hybrid_camview2/results_hybrid_camview2.txt')
hybrid_camview3_path= os.path.join(root_dir, 'hybrid_camview3/results_hybrid_camview3.txt')

hybrid_camview_3_200hz_path= os.path.join(root_dir, 'hybrid_camview3_200hz/hybrid_camview3_200hz.txt')

snn_camview2_spikes = get_firing_rates(snn_camview2_path)
snn_camview3_spikes = get_firing_rates(snn_camview3_path)

snn_camview2_spikes_sorted = np.zeros((num_layers, num_bins))
snn_camview3_spikes_sorted = np.zeros((num_layers, num_bins))
for idx, layer_name in enumerate(layer_names_snn):
    snn_camview2_spikes_sorted[idx] = snn_camview2_spikes[layer_name]
    snn_camview3_spikes_sorted[idx] = snn_camview3_spikes[layer_name]

hybrid_camview2_spikes = get_firing_rates(hybrid_camview2_path)
hybrid_camview3_spikes = get_firing_rates(hybrid_camview3_path)

hybrid_camview2_spikes_sorted = np.zeros((num_layers, num_bins))
hybrid_camview3_spikes_sorted = np.zeros((num_layers, num_bins))
for idx, layer_name in enumerate(layer_names):
    hybrid_camview2_spikes_sorted[idx] = hybrid_camview2_spikes[layer_name]
    hybrid_camview3_spikes_sorted[idx] = hybrid_camview3_spikes[layer_name]

hybrid_camview3_200hz_spikes = get_firing_rates(hybrid_camview_3_200hz_path, num_bins=20)
hybrid_camview3_200hz_spikes_sorted = np.zeros((num_layers, 20))
for idx, layer_name in enumerate(layer_names):
    hybrid_camview3_200hz_spikes_sorted[idx,:] = hybrid_camview3_200hz_spikes[layer_name]

num_mops = [
104.86,
838.86,
838.86,
838.86,
603.98,
603.98,
6710.89,
6710.89,
6710.89]

energy_time_hybrid_camview3_200hz = np.zeros(20)

for idx in range(20):
    energy_time_hybrid_camview3_200hz[idx] = np.sum(hybrid_camview3_200hz_spikes_sorted[:, idx]*num_mops)*0.38/1e3 
