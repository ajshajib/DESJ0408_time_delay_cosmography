import numpy as np
import pickle
import sys

from output_class import ModelOutput

run_name = str(sys.argv[1])
run_type = str(sys.argv[2])

output_file = run_name+'_mod_out.txt'

start_index = int(sys.argv[3])
num_compute = int(sys.argv[4])

print(output_file, run_type, start_index)
base_dir = '../model_posteriors/lens_models/' #'/u/flashscratch/a/ajshajib/modified_outputs/'

out_dir = '../model_posteriors/velocity_dispersion/' #'/u/flashscratch/a/ajshajib/velocity_dispersions/'

output = ModelOutput(base_dir+output_file, run_type, is_test=False)
print('loaded {}'.format(base_dir+output_file))
print('model type: {}'.format(run_type))


output.compute_model_velocity_dispersion(start_index=start_index,
                                         num_compute=num_compute,
                                         print_step=5)
                                


np.savetxt(out_dir+'vd_hern_{}_'.format(start_index)+output_file, output.model_velocity_dispersion)
np.savetxt(out_dir+'rani_hern_{}_'.format(start_index)+output_file, output.r_ani)

print('finished computing velocity dispersions', output.model_velocity_dispersion.shape)
