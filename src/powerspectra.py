"""

Computes the power spectra of the true, predicted, and benchmark density fields. 

"""

import Pk_library as PKL
import numpy as np
import h5py


test_cube = np.load('/projects/QUIJOTE/Noah/dm2gal/dat/processed/test_cube_target.npy')
pred_cube = np.load('/projects/QUIJOTE/Noah/dm2gal/dat/processed/test_cube_version_5167072_prediction.npy')
benchmark_cube = np.load('/projects/QUIJOTE/Noah/dm2gal/dat/processed/benchmark_cube.npy')


BoxSize = 31.82373046875 #Size of the density field in Mpc/h


axis = 0
MAS     = None
threads = 32




Pk = PKL.Pk(test_cube, BoxSize, axis, MAS, threads)
k       = Pk.k3D
Pk0     = Pk.Pk[:,0] #monopole
np.save('../dat/analysis/target_k_values.npy',k)
np.save('../dat/analysis/target_Pk0_values.npy',Pk0)


Pk = PKL.Pk(pred_cube, BoxSize, axis, MAS, threads)

# 3D P(k)
k       = Pk.k3D
Pk0     = Pk.Pk[:,0] #monopole
np.save('../dat/analysis/'+'pred_k_values.npy',k)
np.save('../dat/analysis/'+'pred_Pk0_values.npy',Pk0)


Pk = PKL.Pk(benchmark_cube, BoxSize, axis, MAS, threads)

# 3D P(k)
k       = Pk.k3D
Pk0     = Pk.Pk[:,0] #monopole
np.save('../dat/analysis/'+'hod_k_values.npy',k)
np.save('../dat/analysis/'+'hod_Pk0_values.npy',Pk0)


print("Done! ")


