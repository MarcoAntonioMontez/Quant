import sys,os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import random
from modules import ga


arr_row = [ 0.268,  0.53,   0.797, 43.62,  40.107, 28.944, -0.323,  0.719,  0.188,  0.99,
  0.231, -0.244, -0.35,  -0.436,  5.62,   3.964,  8.081]

f_min = 0.5
f_max = 10
f_range = (f_min, f_max)
# period_range = (i_min, i_max)
unit_range = (0, 1)
double_range = (-0.5, 1.5)
volume_limit_range = (-0.5, 0.5)
total_buy_limit_range = (0, 1)
period_range = (10, 50)

weight_names = ['weight_vol_1', 'weight_vol_2', 'weight_vol_3']
exit_names = ['weight_exit_1', 'weight_exit_2', 'weight_exit_3']

master_genes = []
master_genes.append(ga.master_gene("weight_exit_1", 0, 'float', unit_range))
master_genes.append(ga.master_gene("weight_exit_2", 0, 'float', unit_range))
master_genes.append(ga.master_gene("weight_exit_3", 0, 'float', unit_range))
master_genes.append(ga.master_gene("exit_ind_1_param", 0, 'float', period_range))
master_genes.append(ga.master_gene("exit_ind_2_param", 0, 'float', period_range))
master_genes.append(ga.master_gene("exit_ind_3_param", 0, 'float', period_range))
master_genes.append(ga.master_gene("confirmation_total_buy_limit", 0, 'float', double_range))

master_genes.append(ga.master_gene("weight_vol_1", 0, 'float', unit_range))
master_genes.append(ga.master_gene("weight_vol_2", 0, 'float', unit_range))
master_genes.append(ga.master_gene("weight_vol_3", 0, 'float', unit_range))
master_genes.append(ga.master_gene("buy_limit_vol_1", 0, 'float', volume_limit_range))
master_genes.append(ga.master_gene("buy_limit_vol_2", 0, 'float', volume_limit_range))
master_genes.append(ga.master_gene("buy_limit_vol_3", 0, 'float', volume_limit_range))
master_genes.append(ga.master_gene("volume_total_buy_limit", 0, 'float', double_range))

master_genes.append(ga.master_gene("stop_loss_parameter", 0, 'float', f_range))
master_genes.append(ga.master_gene("trailing_stop_parameter", 0, 'float', f_range))
master_genes.append(ga.master_gene("take_profit_parameter", 0, 'float', f_range))

def decoder(arr_row, master_list):
    l = len(arr_row)
    if l != len(master_list):
        raise Exception("Error! Array length must be the same as gene_list length!")
    decoded_params = {}
    for i in range(0, l):
        mgene = master_list[i]
        print(mgene)
        if mgene['type'] == 'int':
            key = mgene["encoding"][int(arr_row[i])]
            decoded_params[mgene['name']] = key
        else:
            decoded_params[mgene['name']] = arr_row[i]
    return decoded_params

print(arr_row)
print(decoder(arr_row,master_genes))