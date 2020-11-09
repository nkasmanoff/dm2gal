"""

Computes the bispectra of the true, predicted, and benchmark density fields. 

"""

import Pk_library as PKL
import numpy as np
import h5py



test_cube = np.load('../dat/processed/test_cube_target.npy')
pred_cube = np.load('../dat/processed/test_cube_final_prediction.npy')
benchmark_cube = np.load('../dat/processed/central-prediction/benchmark_cube.npy')


BoxSize = 31.82373046875 #Size of the density field in Mpc/h


axis = 0
k1      = 0.5    #h/Mpc
k2      = 0.6    #h/Mpc
MAS     = 'CIC'
threads = 32
theta   = np.linspace(0, np.pi, 25) #array with the angles between k1 and k2



# compute true bispectrum
Bk_target = PKL.Bk(test_cube, BoxSize, k1, k2, theta, MAS, threads)

Bk = Bk_target.B     #bispectrum
Qk = Bk_target.Q     #reduced bispectrum
k  = Bk_target.k_all #k-bins for power spectrum
Pk = Bk_target.Pk    #power spectrum


# save bispectrum and k
np.save('../dat/processed/'+'target_theta_values.npy',theta)
np.save('../dat/processed/'+'target_Bk_values.npy',Bk)

# compute dm2gak bispectrum

# compute bispectrum
Bk_pred = PKL.Bk(pred_cube, BoxSize, k1, k2, theta, MAS, threads)

Bk = Bk_pred.B     #bispectrum
Qk = Bk_pred.Q     #reduced bispectrum
k  = Bk_pred.k_all #k-bins for power spectrum
Pk = Bk_pred.Pk    #power spectrum

np.save('../dat/processed/'+'pred_theta_values.npy',theta)
np.save('../dat/processed/'+'pred_Bk_values.npy',Bk)



# compute benchmark bispectrum
Bk_benchmark = PKL.Bk(benchmark_cube, BoxSize, k1, k2, theta, MAS, threads)

Bk = Bk_benchmark.B     #bispectrum
Qk = Bk_benchmark.Q     #reduced bispectrum
k  = Bk_benchmark.k_all #k-bins for power spectrum
Pk = Bk_benchmark.Pk    #power spectrum

np.save('../dat/processed/'+'benchmark_theta_values.npy',theta)
np.save('../dat/processed/'+'benchmark_Bk_values.npy',Bk)
