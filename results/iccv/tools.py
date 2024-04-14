import matplotlib.pyplot as plt
import numpy as np
import pdb

def get_firing_rates(txt_path, num_layers=9, num_bins=10):
    # pdb.set_trace()
    path_dir = txt_path
    spike_firing_rates = np.zeros((num_layers, num_bins))
    spike_activity_dict = {}
    with open(path_dir, 'r') as f: 
        count = 0
        for line in f:
            if 'total' in line:
                break
            else: 
                if '[' in line: 
                    layer_name = line.split(':')[0]
                    vals = line.split('[')[1].split(' ')
                    while '' in vals:
                        vals.remove('')
                    for val_id, val in enumerate(vals):
                        while '' in vals:
                            vals.remove('')
                        spike_firing_rates[count, val_id] = np.double(val)
                        prev_val = val_id + 1
                elif ']' in line:
                    vals = line.split(']')[0].split(' ')
                    while '' in vals:
                        vals.remove('')
                    for val_id, val in enumerate(vals):
                        spike_firing_rates[count, val_id + prev_val] = np.double(val)
                    spike_activity_dict[layer_name] = spike_firing_rates[count]
                    count += 1
                else: 
                    vals = line.split(' ')[1:]
                    while '' in vals:
                        vals.remove('')
                    for val_id, val in enumerate(vals):
                        # pdb.set_trace()
                        if val == '': 
                            continue
                        while '' in vals:
                            vals.remove('')
                        spike_firing_rates[count, prev_val + val_id] = np.double(val)
                    prev_val += val_id + 1
                    print()

    return spike_activity_dict         
    # return spike_firing_rates.sum(axis=0)/num_layers, spike_firing_rates.sum(axis=1)/num_bins

# def avg_spike_firing_rate_across_layers(txt_path, num_layers, num_bins):
#     path_dir = txt_path
#     spike_firing_rates = np.zeros((num_layers, num_bins))
#     with open(path_dir, 'r') as f: 
#         # pdb.set_trace()
#         count = 0
#         for line in f:
#             if 'total' in line:
#                 break
#             else: 
#                 if '[' in line: 
#                     # pdb.set_trace()
#                     vals = line.split('[')[1].split(' ')
#                     for val_id, val in enumerate(vals):
#                         # pdb.set_trace()
#                         while '' in vals:
#                             vals.remove('')
#                         spike_firing_rates[count, val_id] = np.double(val)
#                         prev_val = val_id + 1
#                 elif ']' in line:
#                     # pdb.set_trace()
#                     vals = line.split(']')[0].split(' ')
#                     while '' in vals:
#                         vals.remove('')
#                     for val_id, val in enumerate(vals):
#                         # pdb.set_trace()
#                         spike_firing_rates[count, val_id + prev_val] = np.double(val)
#                     # lines[count].append()
#                     count += 1
#     return spike_firing_rates.sum(axis=1)/num_bins
