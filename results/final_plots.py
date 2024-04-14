import matplotlib.pyplot as plt
import numpy as np
import pdb

def avg_spike_firing_rate(txt_filename, num_layers, num_bins):
    txt_path='/home/asude/thesis/master_thesis_asude_aydin/rpg_e2vid/all_results_comparison/txt_files/'
    path_dir = txt_path + txt_filename
    spike_firing_rates = np.zeros((num_layers, num_bins))
    with open(path_dir, 'r') as f: 
        # pdb.set_trace()
        count = 0
        for line in f:
            if 'total' in line:
                break
            else: 
                if '[' in line: 
                    # pdb.set_trace()
                    vals = line.split('[')[1].split(' ')
                    for val_id, val in enumerate(vals):
                        # pdb.set_trace()
                        while '' in vals:
                            vals.remove('')
                        spike_firing_rates[count, val_id] = np.double(val)
                        prev_val = val_id + 1
                elif ']' in line:
                    # pdb.set_trace()
                    vals = line.split(']')[0].split(' ')
                    while '' in vals:
                        vals.remove('')
                    for val_id, val in enumerate(vals):
                        # pdb.set_trace()
                        spike_firing_rates[count, val_id + prev_val] = np.double(val)
                    # lines[count].append()
                    count += 1
    return spike_firing_rates.sum(axis=0)/num_layers

def avg_spike_firing_rate_across_layers(txt_filename, num_layers, num_bins):
    txt_path='/home/asude/thesis/master_thesis_asude_aydin/rpg_e2vid/all_results_comparison/txt_files/'
    path_dir = txt_path + txt_filename
    spike_firing_rates = np.zeros((num_layers, num_bins))
    with open(path_dir, 'r') as f: 
        # pdb.set_trace()
        count = 0
        for line in f:
            if 'total' in line:
                break
            else: 
                if '[' in line: 
                    # pdb.set_trace()
                    vals = line.split('[')[1].split(' ')
                    for val_id, val in enumerate(vals):
                        # pdb.set_trace()
                        while '' in vals:
                            vals.remove('')
                        spike_firing_rates[count, val_id] = np.double(val)
                        prev_val = val_id + 1
                elif ']' in line:
                    # pdb.set_trace()
                    vals = line.split(']')[0].split(' ')
                    while '' in vals:
                        vals.remove('')
                    for val_id, val in enumerate(vals):
                        # pdb.set_trace()
                        spike_firing_rates[count, val_id + prev_val] = np.double(val)
                    # lines[count].append()
                    count += 1
    return spike_firing_rates.sum(axis=1)/num_bins


# CNN SCORES
cnn_100hz = [5.71898548, 5.72030754, 5.7230106, 5.72487041, 5.72551493, 5.72608551, 5.72719143, 5.72947849, 5.7295493, 5.72798925]
cnn_10hz = [5.8369944, 5.943272,  6.084734,  6.2498148, 6.4313217, 6.6234018, 6.8201562, 7.0206374, 7.2228734, 7.4236708]

cnn_10hz_camview2 = [4.73680406, 4.85949938, 5.01783954, 5.19875879, 5.39451882, 5.59979851, 5.81017615, 6.02164049, 6.23675615, 6.44970663]
# SNN SCORES

# NEW: CONV+BN
snn_delta_pred_w_init = [5.89546237, 5.94492591, 6.00495231, 6.03628731, 5.97243063, 5.92162749, 5.88303144, 5.85002134, 5.88546653, 5.91659745]
conv_bn = [5.89546237, 5.94492591, 6.00495231, 6.03628731, 5.97243063, 5.92162749, 5.88303144, 5.85002134, 5.88546653, 5.91659745]
snn_scratch_state_init = [6.81491914, 6.56586276, 6.30473211, 6.36845792, 6.41245106, 6.39685706, 6.53499586, 6.75310335, 7.00861474, 7.26425049]
conv_bn_camview2 = [4.82870833, 4.88269076, 4.96948507, 5.05115678, 5.03071034, 4.98318214, 4.94123742, 4.94343562, 4.99423122, 5.09872133]


snn_delta_pred_no_init = [5.84024628, 5.98058399, 6.09965859, 6.2939205,  6.55834856, 6.97069533, 7.50676323, 7.96861704, 8.47129119, 9.11456918]
# snn_delta_pred_w_init = [5.91961313, 6.36662062, 6.36038539, 6.14043627, 6.03371847, 5.95041738, 5.89417316, 5.85326382, 5.84941175, 5.85943476]
# snn_delta_pred_w_init_075 = [6.00819547, 6.23472485, 6.23606239, 6.18439659, 6.07139571, 5.96441823, 5.87817634, 5.83045901, 5.81966804, 5.87875658]
snn_pred_from_scratch = [38.73337764, 24.81840765, 18.35099915, 15.07266592, 13.3075388,  12.13983863, 11.22621023, 10.65870277, 10.19260069, 9.8089811]
# snn_scratch_state_init = [7.47700502, 6.87729605, 6.4957692,  6.27334363, 6.38499859, 6.31905018, 6.21772382, 6.22667113, 6.24693698, 6.48525963] # THIS WAS WRONG I SHOULDVE TAKEN THE 5TH EPOCH INSTEAD OF THE 3RD EPOCH
snn_pred_from_scratch_20bins_30inference = [39.21959547, 25.53474767, 22.14910935, 20.24076663, 17.64263082, 14.1742866, 11.74035937, 10.48165559, 9.48790984, 8.84758562, 8.38499097, 7.99383466, 7.64490268, 7.45237584, 7.24933875, 7.10052377, 6.97984767, 6.89080662, 6.79196136, 6.75253232, 6.67029131, 6.62641788, 6.56759628, 6.52613918, 6.50375804, 6.46894203, 6.44545973, 6.41967032, 6.41101941, 6.39474232]
snn_pred_from_scratch_20bins = [39.25241749, 25.56891131, 22.18399034, 20.27228184, 17.67057066, 14.1942645, 11.76014858, 10.49899312, 9.51523163, 8.87998171, 8.38714296, 8.01238841, 7.65527013, 7.42249156, 7.26144505, 7.11425848, 6.98793607, 6.88959559, 6.79797155, 6.76493241]
# TODO UPDATE 
# snn_delta_pred_w_init_20bins_inference = [7.75159619, 8.316963, 8.33225693, 8.08772908, 7.8574559, 7.48804999, 7.11283465, 6.77988684, 6.49495866, 6.29718809, 6.18258729, 6.15854534, 6.21385123, 6.36191577, 6.61219077, 7.00643342, 7.59994989, 8.43149679, 9.68527585, 11.44049154]
snn_delta_pred_w_init_25bins_inference = [ 5.9958272, 6.52917748, 6.50334569, 6.21209095, 6.08583092, 6.00204498, 5.93761843, 5.90536015, 5.88966495, 5.9007442, 5.94467408, 6.03725636, 6.17201287, 6.3585478, 6.63834887, 7.07431905, 7.69559347, 8.56153606,
  9.85387032, 11.6043932, 14.18622089, 17.16209002, 20.33359859, 22.85052966,
 23.90876309]
snn_pred_from_scratch_30bins_inference = [38.63520305, 24.71858671, 18.30399325, 15.0053174, 13.25854479, 12.10310891,
 11.17659042, 10.60235092, 10.12351106,  9.75471348,  9.42846062, 9.15785624,
  8.87765977,  8.75335079,  8.66289325,  8.56827464,  8.51294227,  8.42841536,
  8.34629116,  8.3147717,   8.31620075,  8.28685319,  8.29816349,  8.28145128,
  8.20932362,  8.20530341,  8.17723467,  8.15590974,  8.14598949,  8.13786156]

hybrid_200hz_const_count_camview3_conv_bn = [5.10256746, 5.21511778, 5.25212598, 5.21126488, 5.15107152, 5.11515455, 5.11086659, 5.12886008, 5.18148671, 5.27569424]
#  24.76115856, 25.1485267,  25.54631249, 25.52584182, 25.68912462]
# RNN SCORES
rnn_from_scratch = [15.27749962, 9.896231, 8.7463126, 7.90452345, 7.16394268, 6.68080832, 6.37496316, 6.15290571, 6.02771848, 5.9305618]
rnn_delta_w_state_init = [5.80346743, 5.73680218, 5.70243484, 5.61908887, 5.61026066, 5.59700109, 5.59375573, 5.55951449, 5.5345149, 5.52215443]
# rnn_delta_w_state_init = [5.73802464, 5.7034841, 5.62689712, 5.5785386, 5.57675796, 5.60276229, 5.59268037, 5.54780158, 5.53575863, 5.52508598]
rnn_delta_no_state_init = [5.76139193, 5.79011066, 5.81725045, 5.84343135, 5.896066, 5.94486489, 5.93200202, 5.91744941, 5.88399242, 5.84361393]
# 3D SCORES NEW
scores_3d = [71.15073557, 71.04888937, 70.74968781, 71.01174108, 72.0735642,  74.00583634, 75.90760842, 77.87819389, 80.5783625,  84.349355]
# scores_3d = [70.85108674, 73.59204391, 72.98330238, 71.59833919, 72.41241222, 73.30365996, 75.27380291, 77.42522244, 79.62591609, 82.3043995] # mean 74.937018534000018

snn_blue_hex = '#007aff'
cnn_blue_hex = '#03468F'
rnn_green_hex = '#007355'
snn_state_init_hex = '#f65353'
rnn_state_init_hex = '#f59a23'

plt.rcParams.update({'font.family':'serif',
                     'font.size': 15})

# plt.figure(figsize=(9, 6))
# plt.plot(conv_bn_camview2, color=snn_blue_hex,  marker='^', linestyle='--',markersize = 10, linewidth=2)
# plt.plot(cnn_10hz_camview2, color = cnn_blue_hex,  marker='o',markersize = 10, linewidth=2)
# plt.legend(['Hybrid CNN - SNN, camview2', 'CNN 10Hz'])
# plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
# plt.ylabel('MPJPE (2D)')
# plt.xlabel('# of Time Steps')
# plt.savefig('hybrid_cnn_snn_camview2.pdf')
# plt.savefig('hybrid_cnn_snn_camview2.png', dpi=400)

# plt.figure(figsize=(9, 6))
# plt.plot(cnn_10hz, color = cnn_blue_hex,  marker='o',markersize = 10, linewidth=2)
# plt.plot(conv_bn, color = snn_state_init_hex,  marker='^', linestyle='--',markersize = 10, linewidth=2)
# plt.legend(['CNN 10Hz', 'Hybrid CNN - SNN, camview3'])
# plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
# plt.xlabel('# of Time Steps')
# plt.ylabel('MPJPE (2D)')
# plt.savefig('hybrid_cnn_snn_camview3.pdf')
# plt.savefig('hybrid_cnn_snn_camview3.png', dpi=400)

plt.figure(figsize=(9, 6))
plt.plot(snn_pred_from_scratch, color=snn_blue_hex, marker = 'o', markersize = 10, linewidth=2,)
plt.plot(snn_scratch_state_init, color=snn_state_init_hex, marker='o', markersize = 10, linewidth=2,)
plt.plot(snn_delta_pred_no_init, color=snn_blue_hex,  marker='^', linestyle='--', markersize = 10, linewidth=2,)
plt.plot(snn_delta_pred_w_init, color=snn_state_init_hex,  marker='^', linestyle='--', markersize = 10, linewidth=2,)
plt.yscale('log')
plt.legend(['Scratch (A)', 'Scratch + State Init (B)', 'Delta (C)', 'Delta + State Init (D - Ours)'])
plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1), fontfamily='serif')
plt.xlabel('# of Time Steps [T=10 ms]')
plt.ylabel('MPJPE (2D)')
plt.tight_layout()
# plt.savefig('hybrid_ablation_plot.png',dpi=400)
plt.savefig('hybrid_ablation_plot.pdf')

plt.figure(figsize=(9, 6))
plt.plot(snn_pred_from_scratch, color=snn_blue_hex, marker = 'o', markersize = 10, linewidth=2,)
# plt.plot(snn_scratch_state_init, color=snn_state_init_hex, marker='o', markersize = 10, linewidth=2,)
# plt.plot(snn_delta_pred_no_init, color=snn_blue_hex,  marker='^', linestyle='--', markersize = 10, linewidth=2,)
# plt.plot(snn_delta_pred_w_init, color=snn_state_init_hex,  marker='^', linestyle='--', markersize = 10, linewidth=2,)
plt.yscale('log')
plt.legend(['Scratch (A)'])
plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1), fontfamily='serif')
plt.xlabel('# of Time Steps [T=10 ms]')
plt.ylabel('MPJPE (2D)')
plt.tight_layout()
# plt.savefig('hybrid_ablation_plot.png',dpi=400)
plt.savefig('hybrid_ablation_plot1.pdf')

plt.figure(figsize=(9, 6))
plt.plot(snn_pred_from_scratch, color=snn_blue_hex, marker = 'o', markersize = 10, linewidth=2,)
# plt.plot(snn_scratch_state_init, color=snn_state_init_hex, marker='o', markersize = 10, linewidth=2,)
# plt.plot(snn_delta_pred_no_init, color=snn_blue_hex,  marker='^', linestyle='--', markersize = 10, linewidth=2,)
# plt.plot(snn_delta_pred_w_init, color=snn_state_init_hex,  marker='^', linestyle='--', markersize = 10, linewidth=2,)
# plt.yscale('log')
plt.legend(['SNN'])
plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1), fontfamily='serif')
plt.xlabel('# of Time Steps')
plt.ylabel('Score')
plt.tight_layout()

for pos in ['top', 'right']:
    plt.gca().spines[pos].set_visible(False)
# plt.savefig('hybrid_ablation_plot.png',dpi=400)
plt.savefig('snn_latency.pdf')

plt.figure(figsize=(9, 6))
plt.plot(snn_pred_from_scratch, color=snn_blue_hex, marker = 'o', markersize = 10, linewidth=2,)
plt.plot(snn_scratch_state_init, color=snn_state_init_hex, marker='o', markersize = 10, linewidth=2,)
# plt.plot(snn_delta_pred_no_init, color=snn_blue_hex,  marker='^', linestyle='--', markersize = 10, linewidth=2,)
# plt.plot(snn_delta_pred_w_init, color=snn_state_init_hex,  marker='^', linestyle='--', markersize = 10, linewidth=2,)
plt.yscale('log')
plt.legend(['Scratch (A)', 'Scratch + State Init (B)'])
plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1), fontfamily='serif')
plt.xlabel('# of Time Steps [T=10 ms]')
plt.ylabel('MPJPE (2D)')
plt.tight_layout()
# plt.savefig('hybrid_ablation_plot.png',dpi=400)
plt.savefig('hybrid_ablation_plot2.pdf')

plt.figure(figsize=(9, 6))
plt.plot(snn_pred_from_scratch, color=snn_blue_hex, marker = 'o', markersize = 10, linewidth=2,)
plt.plot(snn_scratch_state_init, color=snn_state_init_hex, marker='o', markersize = 10, linewidth=2,)
plt.plot(snn_delta_pred_no_init, color=snn_blue_hex,  marker='^', linestyle='--', markersize = 10, linewidth=2,)
# plt.plot(snn_delta_pred_w_init, color=snn_state_init_hex,  marker='^', linestyle='--', markersize = 10, linewidth=2,)
plt.yscale('log')
plt.legend(['Scratch (A)', 'Scratch + State Init (B)', 'Delta (C)'])
plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1), fontfamily='serif')
plt.xlabel('# of Time Steps [T=10 ms]')
plt.ylabel('MPJPE (2D)')
plt.tight_layout()
# plt.savefig('hybrid_ablation_plot.png',dpi=400)
plt.savefig('hybrid_ablation_plot3.pdf')

# This plot runs snn for 30 bins and comares effect of training for 10 bins vs 20 bins.
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }
plt.figure(figsize=(10, 6))
plt.hlines(y = 6, xmin = 0, xmax = 29, color='black', linestyle=':', linewidth=2) # state init -> hybrid blue -> delta -> --
plt.plot(snn_delta_pred_w_init, color=snn_state_init_hex, marker='^', linestyle='--', markersize = 10, linewidth=2) # state init -> hybrid blue -> delta -> --
markers_on = np.arange(0,20,1)
plt.plot(snn_pred_from_scratch_20bins_30inference, color=snn_blue_hex, marker='o', markersize = 10, linewidth=2, markevery=markers_on)
plt.legend(['Delta + State Init', 'Scratch for 30 Steps', 'Horizontal Line at Score 6' ])
plt.xticks(np.arange(0, 30, 1), np.arange(1, 31, 1))
plt.yscale('log')
plt.locator_params(axis='x', nbins=18)
plt.xlabel('# of Time Steps [T=10 ms]')
plt.ylabel('MPJPE (2D)')
plt.tight_layout()
# plt.savefig('snn_scratch_30bins_inference.png', dpi=400)
plt.savefig('snn_scratch_30bins_inference.pdf')

plt.figure(figsize=(9, 6))
plt.plot(snn_delta_pred_no_init, color=snn_blue_hex,  marker='^', linestyle='--',markersize = 10, linewidth=2)
plt.plot(cnn_10hz, color = cnn_blue_hex,  marker='o',markersize = 10, linewidth=2)
plt.legend(['Delta', 'CNN 10Hz'])
plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
plt.ylabel('MPJPE (2D)')
plt.xlabel('# of Time Steps')
plt.savefig('delta_no_init_cnn_10hz_comp.pdf')
# plt.savefig('delta_no_init_cnn_10hz_comp.png', dpi=400)

# SHOW SNN vs ANN COMPARISON 
plt.figure(figsize=(9, 6))
plt.plot(rnn_from_scratch, color = rnn_green_hex,  marker='o',markersize = 10, linewidth=2)
plt.plot(rnn_delta_no_state_init, color = rnn_green_hex,  marker='^', linestyle='--',markersize = 10, linewidth=2)
plt.plot(rnn_delta_w_state_init, color=rnn_state_init_hex, marker='^', linestyle='--',markersize = 10, linewidth=2)
plt.legend(['RNN Scratch', 'RNN Delta', 'RNN Delta + State Init'])
plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
plt.xlabel('# of Time Steps')
plt.ylabel('MPJPE (2D)')
plt.savefig('rnn_comp.pdf')
plt.savefig('rnn_comp.png', dpi=400)


# SHOW SNN vs ANN COMPARISON ZOOMED -> TODO: UPDATE RNN DELTA PRED RESULT 
plt.figure(figsize=(9, 6))
plt.plot(cnn_100hz, color = cnn_blue_hex,  marker='s',markersize = 10, linewidth=2)
plt.plot(cnn_10hz, color = cnn_blue_hex,  marker='o',markersize = 10, linewidth=2)
plt.plot(conv_bn, color = snn_state_init_hex,  marker='^', linestyle='--',markersize = 10, linewidth=2)
# plt.plot(hybrid_200hz_const_count_camview3_conv_bn, color = 'r', marker = '3', markersize=10, linewidth=2)
# plt.plot(snn_delta_pred_w_init, color = snn_state_init_hex,  marker='^', linestyle='--')
# plt.plot(rnn_from_scratch, color = rnn_green_hex,  marker='o')
plt.plot(rnn_delta_w_state_init, color=rnn_state_init_hex, marker='^', linestyle='--',markersize = 10, linewidth=2)
# add rnn state init
plt.legend(['CNN 100Hz', 'CNN 10Hz', 'SNN Delta + State Init', 'SNN Delta + State Init 200 Hz', 'RNN Delta + State Init'])
plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
plt.xlabel('# of Time Steps')
plt.ylabel('MPJPE (2D)')
# plt.title('Accuracy Comparison: SNN State Init vs. ANNs Zoomed')
plt.savefig('snn_vs_ann_comp_zoomed.png',dpi=400)
plt.savefig('snn_vs_ann_comp_zoomed.pdf')

plt.figure(figsize=(9, 6))
plt.plot(cnn_100hz, color = cnn_blue_hex,  marker='s',markersize = 10, linewidth=2)
plt.plot(cnn_10hz, color = cnn_blue_hex,  marker='o',markersize = 10, linewidth=2)
plt.plot(conv_bn, color = snn_state_init_hex,  marker='^', linestyle='--',markersize = 10, linewidth=2)
# plt.plot(snn_delta_pred_w_init, color = snn_state_init_hex,  marker='^', linestyle='--')
# plt.plot(rnn_from_scratch, color = rnn_green_hex,  marker='o')
# plt.plot(rnn_delta_w_state_init, color=rnn_state_init_hex, marker='^', linestyle='--',markersize = 10, linewidth=2)
# add rnn state init
plt.legend(['CNN 100Hz', 'CNN 10Hz', 'Delta + State Init (Ours)'])
plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
plt.xlabel('# of Time Steps')
plt.ylabel('MPJPE (2D)')
# plt.title('Accuracy Comparison: SNN State Init vs. ANNs Zoomed')
plt.savefig('snn_vs_ann_comp_no_rnn.pdf')


# ACCURACY VS ENERGY
plt.figure()
plt.plot(101.3*10, np.mean(cnn_100hz), color = cnn_blue_hex,  marker='s', markersize=20)
plt.plot(101.3, np.mean(cnn_10hz), color = cnn_blue_hex,  marker='o', markersize=20)
plt.plot(101.3 + 8.72 + 5.86, np.mean(snn_delta_pred_w_init), color = snn_state_init_hex,  marker='^', markersize=20)
# plt.plot(78*10, np.mean(rnn_from_scratch), color = rnn_green_hex,  marker='o', markersize=15)
plt.plot(78*10 + 101.3, np.mean(rnn_delta_w_state_init), color=rnn_state_init_hex, marker='^', markersize=20)
plt.xscale('log')
plt.legend(['CNN 100Hz', 'CNN 10Hz', 'SNN Delta + State Init', 'RNN Delta + State Init'])
# plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
plt.xlabel('Energy Cost (mJ)')
plt.ylabel('MPJPE (2D)')
# plt.title('Energy Costs vs. Accuracy')
plt.tight_layout()
plt.savefig('accuracy_vs_energy.png', dpi=400)
plt.savefig('accuracy_vs_energy.pdf')

# ACCURACY VS ENERGY
plt.figure()
plt.plot(101.3*10, np.mean(cnn_100hz), color = cnn_blue_hex,  marker='s', markersize=20)
plt.plot(101.3, np.mean(cnn_10hz), color = cnn_blue_hex,  marker='o', markersize=20)
plt.plot(101.3 + 8.72 + 5.86, np.mean(snn_delta_pred_w_init), color = snn_state_init_hex,  marker='^', markersize=20)
# plt.plot(78*10, np.mean(rnn_from_scratch), color = rnn_green_hex,  marker='o', markersize=15)
# plt.plot(78*10 + 101.3, np.mean(rnn_delta_w_state_init), color=rnn_state_init_hex, marker='^', markersize=20)
plt.xscale('log')
plt.legend(['CNN 100Hz', 'CNN 10Hz', 'Delta + State Init (Ours)'])
# plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
plt.xlabel('Energy Cost (mJ)')
plt.ylabel('MPJPE (2D)')
# plt.title('Energy Costs vs. Accuracy')
plt.tight_layout()
plt.savefig('accuracy_vs_energy_no_rnn.pdf')


# plt.figure()
labels = 'State Init', 'CNN', 'SNN'
sizes = [8, 101.3, 5.86]
explode = (0, 0, 0.15)  # only "explode" the 3rd slice 
fig1, ax1 = plt.subplots()
_,_, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors = [snn_blue_hex, cnn_blue_hex, snn_state_init_hex])

centre_circle = plt.Circle((0,0),0.20,fc='white')
fig1.gca().add_artist(centre_circle)

for autotext in autotexts:
    autotext.set_color('white')
    # autotext.
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.title('Energy Cost Distribution at 100Hz')
plt.tight_layout()
plt.savefig('pie_chart.png')


# RUN SNN FOR LONGER -> TODO: UPDATE
plt.figure()
# plt.plot(snn_delta_pred_w_init)
plt.plot(snn_delta_pred_w_init_25bins_inference, color=snn_state_init_hex, marker='^', linestyle='--')
plt.legend(['SNN Delta + State Init'])
plt.xticks(np.arange(0, 25, 1), np.arange(1, 26, 1))
plt.xlabel('# of Time Steps')
plt.ylabel('MPJPE (2D)')
plt.title('Inference for 30 time steps')
plt.savefig('snn_30bins_inference.pdf')
# pdb.set_trace()

# This plot runs snn for 30 bins and comares effect of training for 10 bins vs 20 bins.
plt.figure(figsize=(10, 6))
# plt.plot(snn_pred_from_scratch, color=snn_blue_hex, marker='s')
plt.plot(snn_pred_from_scratch_30bins_inference, color=snn_blue_hex, marker='o')
plt.plot(snn_pred_from_scratch_20bins_30inference, color='#B0E3E6', marker='s')
for val_id, val in enumerate(snn_pred_from_scratch):
    shift = 1.5
    plt.vlines(x = val_id, ymin=(val-shift), ymax=(val+shift), color=snn_blue_hex)
for val_id, val in enumerate(snn_pred_from_scratch_20bins):
    shift = 1.5
    plt.vlines(x = val_id, ymin=(val-shift), ymax=(val+shift), color='#B0E3E6')
plt.legend(['SNN Scratch 10 Bins', 'SNN Scratch 20 Bins'])
plt.xticks(np.arange(0, 30, 1), np.arange(1, 31, 1))
plt.xlabel('# of Time Steps')
plt.ylabel('MPJPE (2D)')
plt.title('Inference for 30 time steps')
# plt.tight_layout()
plt.savefig('snn_scratch_inference_comp.png')
plt.savefig('snn_scratch_inference_comp.pdf')

# This plot compares inference for more time steps.
plt.figure()
plt.plot(snn_pred_from_scratch_20bins, color=snn_blue_hex, marker='s', linestyle='-.')
plt.plot(snn_pred_from_scratch, color=snn_blue_hex, marker = 'o') # no state init -> snn blue -> from scratch 
plt.plot(snn_delta_pred_w_init, color=snn_state_init_hex, marker='^', linestyle='--') # state init -> hybrid blue -> delta -> --
# plt.plot(rnn_from_scratch, color = cnn_blue_hex,  marker='o', linestyle = '--')
plt.legend(['SNN Scratch 20 bins', 'SNN Scratch', 'SNN Delta + State Init'])
plt.xticks(np.arange(0, 20, 1), np.arange(1, 21, 1))
plt.xlabel('# of Time Steps')
plt.ylabel('MPJPE (2D)')
# plt.title('Accuracy Comparisons: SNN Scratch vs. SNN State Init')
plt.savefig('snn_comps.pdf')
# pdb.set_trace()

txt_filename='snn_delta_pred_with_state_init.txt'
txt_filename = 'new/hybrid_cameraview3_conv+bn.txt'
spikes_with_state_init = avg_spike_firing_rate(txt_filename, num_layers=9, num_bins=10)
avg_spikes_with_state_init = np.mean(spikes_with_state_init)
txt_filename='snn_pred_from_scratch.txt'
spikes_from_scratch = avg_spike_firing_rate(txt_filename, num_layers=9, num_bins=10)
avg_spikes_from_scratch = np.mean(spikes_from_scratch)

plt.figure()
plt.plot(spikes_from_scratch, color=snn_blue_hex, marker = 'o')
plt.plot(spikes_with_state_init, color=snn_state_init_hex,  marker='^', linestyle='--')
plt.legend(['Scratch', 'Delta + State Init'])
plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
plt.ylabel('Spike Firing Rate')
plt.xlabel('Time Steps')
# plt.title('SNN Spike Firing Rates')
plt.tight_layout()
plt.savefig('spike_firing_rate.pdf')
plt.savefig('spike_firing_rate.png')

txt_filename='snn_delta_pred_with_state_init.txt'
txt_filename = 'new/hybrid_cameraview3_conv+bn.txt'
spikes_with_state_init = avg_spike_firing_rate_across_layers(txt_filename, num_layers=9, num_bins=10)
avg_spikes_with_state_init = np.mean(spikes_with_state_init)
txt_filename='snn_pred_from_scratch.txt'
spikes_from_scratch = avg_spike_firing_rate_across_layers(txt_filename, num_layers=9, num_bins=10)
avg_spikes_from_scratch = np.mean(spikes_from_scratch)

plt.figure()
plt.plot(spikes_from_scratch, color=snn_blue_hex, marker = 'o')
plt.plot(spikes_with_state_init, color=snn_state_init_hex,  marker='^', linestyle='--')
plt.legend(['Scratch', 'Delta + State Init'])
plt.xticks(np.arange(0, 9, 1), np.arange(1, 10, 1))
plt.ylabel('Spike Firing Rate')
plt.xlabel('Layer Number')
# plt.title('SNN Spike Firing Rates')
plt.tight_layout()
plt.savefig('spike_firing_rate_layers.pdf')
plt.savefig('spike_firing_rate_layers.png')

plt.figure()
# dupl = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
# hz = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0])
hz = np.array([100.0, 200.0])

#74.88
plt.plot(hz, [114, 116], color = '#054907',  marker='^', markersize=15)
#79.63
plt.plot(hz, 11.72*10*np.array([1,2]),  color = '#008000',  marker='s', markersize=15)
plt.plot(hz, 12.1*10*np.array([1,2]),  color = '#15B01A',  marker='s', markersize=15)

# 70.1
# plt.plot(hz, 101.3*10*dupl, color = cnn_blue_hex,  marker='o', markersize=15)
# energy needs to be recalculated 
# plt.plot(11.72*10 + 100, 58.4, np.mean(snn_delta_pred_w_init), color = snn_state_init_hex,  marker='^', markersize=15)
# plt.xscale(' log')
# plt.xticks(dupl, hz)
plt.legend(['Ours', 'DHP19', 'Baldwin et al.'])
# plt.xticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
plt.xlabel('Rate (Hz)')
plt.ylabel('Energy Cost (mJ)')
plt.tight_layout()
plt.savefig('cost_acc_3d_short.pdf')
plt.savefig('cost_acc_3d_short.png')

fig, ax = plt.subplots(figsize=(1.5,5))
# hz = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0])
#74.88
ax.plot(0, 74.88, color = snn_blue_hex,  marker='^', markersize=20)
#79.63
ax.plot(0, 79.63, color = cnn_blue_hex,  marker='s', markersize=20)
# 70.1
ax.plot(0, 70.1, color = cnn_blue_hex,  marker='o', markersize=20)
# plt.legend(['Ours', 'DHP19', 'Our CNN'])
# plt.xlabel('Rate (Hz)')
ax.yaxis.tick_right()
ax.set_ylabel('MPJPE (3D)')
ax.get_xaxis().set_visible(False)
plt.tight_layout()
plt.savefig('acc_3d.pdf')
plt.savefig('acc_3d.png')

plt.close('all')