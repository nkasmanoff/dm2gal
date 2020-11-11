"""

Compute the correlation function of the true, predicted, and benchmark density fields. 

https://github.com/franciscovillaescusa/Pylians3/blob/master/documentation/Documentation.md#CF
"""

import Pk_library as PKL
import numpy as np
import h5py
import sys


test_cube = np.load('../dat/processed/test_cube_target.npy')
pred_cube = np.load('../dat/processed/test_cube_final_prediction.npy')
benchmark_cube = np.load('../dat/processed/benchmark_cube.npy')


BoxSize = 31.82373046875 #Size of the density field in Mpc/h
MAS     = None
threads = 16
axis    = 0


# compute the correlation function
CF     = PKL.Xi(test_cube, BoxSize, MAS, axis, threads)
r      = CF.r3D #radii in Mpc/h
xi0    = CF.xi[:,0]  #correlation function (monopole)
#xi2    = CF.xi[:,1]  #correlation function (quadrupole)
#xi4    = CF.xi[:,2]  #correlation function (hexadecapole)

# save correlation function and r
np.save('../processed/target_r_values.npy',r)
np.save('../processed/target_xi0_values.npy',xi0)

# compute dm2gal corr func
CF     = PKL.Xi(pred_cube, BoxSize, MAS, axis, threads)
r      = CF.r3D #radii in Mpc/h
xi0    = CF.xi[:,0]  #correlation function (monopole)
#xi2    = CF.xi[:,1]  #correlation function (quadrupole)
#xi4    = CF.xi[:,2]  #correlation function (hexadecapole)

# save correlation function and r
np.save('../processed/pred_r_values.npy',r)
np.save('../processed/pred_xi0_values.npy',xi0)


# compute benchmark corr func

CF     = PKL.Xi(benchmark_cube, BoxSize, MAS, axis, threads)
r      = CF.r3D #radii in Mpc/h
xi0    = CF.xi[:,0]  #correlation function (monopole)
#xi2    = CF.xi[:,1]  #correlation function (quadrupole)
#xi4    = CF.xi[:,2]  #correlation function (hexadecapole)

# save correlation function and r
np.save('../processed/benchmark_r_values.npy',r)
np.save('../processed/benchmark_xi0_values.npy',xi0)