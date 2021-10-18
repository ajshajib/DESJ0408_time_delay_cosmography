import numpy as np
import pickle
import sys

from output_class_hern2 import ModelOutput

run_name = str(sys.argv[1])
run_type = str(sys.argv[2])

output_file = run_name+'_mod_out.txt'

start_index = int(sys.argv[3])
num_compute = int(sys.argv[4])

print(output_file, run_type, start_index)
base_dir = '/Users/ajshajib/Research/time_delay_cosmography/J0408/final_nested_samples/modified_outputs/' #'/u/flashscratch/a/ajshajib/modified_outputs/'

out_dir = '/Users/ajshajib/Research/time_delay_cosmography/J0408/vel_dis_test/'

output = ModelOutput(base_dir+output_file, run_type, is_test=False)
print('loaded {}'.format(base_dir+output_file))
print('model type: {}'.format(run_type))

#output.samples_mcmc = output.samples_mcmc[:2]

#output.compute_model_time_delays()
#print('finished computing time delays', output.model_time_delays.shape)


output.compute_model_velocity_dispersion(start_index=start_index,
                                         num_compute=num_compute,
                                         print_step=5)
                                

#np.savetxt(base_dir+'td_{}'.format()+output_file, output.model_time_delays)
np.savetxt(out_dir+'vd_hern_{}_'.format(start_index)+output_file, output.model_velocity_dispersion)
np.savetxt(out_dir+'rani_hern_{}_'.format(start_index)+output_file, output.r_ani)

print('finished computing velocity dispersions', output.model_velocity_dispersion.shape)

#with open(base_dir+'processed_'+output_file, 'wb') as f:
#    pickle.dump(output, f)
