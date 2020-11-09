"""
After training and generating the cosmologically relevant statistics, load in those arrays
and plot and saved.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py


k = np.load('../dat/analysis/target_k_values.npy')
Pk_target  = np.load('../dat/analysis/target_Pk0_values.npy')
Pk_pred = np.load('../dat/analysis/pred_Pk0_values.npy')

Pk_benchmark = np.load('../dat/analysis/benchmark_Pk0_values.npy')


plt.figure(figsize = (7,5))
plt.plot(k[k <= 60],Pk_pred[k <= 60],'--',color='crimson',linewidth=3,label = 'dm2gal')
plt.plot(np.concatenate([k[k<=60][:50][::5],k[k <= 60][50::20]]),np.concatenate([Pk_benchmark[k<=60][:50][::5],Pk_benchmark[k <= 60][50::20]]),'s--',color='limegreen',markersize=3,label = 'HOD')

plt.plot(k[k <= 60],Pk_target[k <= 60],'slateblue',linewidth=3,label = 'Target',alpha=.75)
plt.xscale('log')
#plt.title("TEST",color='white',fontsize=20)
plt.title(r"$k_1 = 3 \frac{h}{Mpc}$, $k_2 = 4 \frac{h}{Mpc}$ ",fontsize=20,color='white',alpha=0)

plt.yscale('log')
plt.xlabel(r'$k$ [${h}{\rm Mpc^{-1}}$]',fontsize = 25)
plt.ylabel(r'$P(k)$  [$(h^{-1}{\rm Mpc})^3$]',fontsize = 25)
plt.legend(fontsize=20)

plt.xticks(fontsize=15)#, rotation=90)
plt.yticks(fontsize=15)#, rotation=90)


plt.tight_layout()

plt.savefig('../bin/power_spectra.png')



r = np.load('../dat/analysis/target_r_values.npy')
xi0_target  = np.load('../dat/analysis/target_xi0_values.npy')
xi0_pred = np.load('../dat/analysis/pred_xi0_values.npy')

xi0_benchmark = np.load('../dat/analysis/benchmark_xi0_values.npy')
plt.figure()
plt.plot(r,xi0_pred,'--',label = 'dm2gal')
plt.plot(r,xi0_benchmark,'--',label = 'benchmark')
plt.plot(r,xi0_target,label = 'target')
#plt.xscale('log')
plt.yscale('log')
plt.xlabel('r',fontsize = 15)
plt.ylabel('xi0(r)',fontsize = 15)
plt.legend()


plt.savefig('../bin/correlation_fn.png')

# Bispectra

Bk_pred = np.load('../dat/analysis/pred_Bk_values.npy')
Bk_benchmark = np.load('../dat/analysis/benchmark_Bk_values.npy')
Bk_target = np.load('../dat/analysis/target_Bk_values.npy')
theta   = np.linspace(0, np.pi, 25) #array with the angles between k1 a$
thetapi = theta / np.pi

plt.figure(figsize = (7,5))


plt.plot(thetapi,Bk_pred,'--',color='crimson',linewidth=3,label = 'dm2gal')
plt.plot(thetapi,Bk_benchmark,'s--',color='limegreen',markersize=3,label = 'HOD')
plt.plot(thetapi,Bk_target,'slateblue',linewidth=3,label = 'Target',alpha=.75)
plt.yscale('log')

plt.legend()
plt.title(r"$k_1$ = 3 ${h^{-1}}{\rm Mpc}$, $k_2$ = 4 ${h^{-1}}{\rm Mpc}$ ",fontsize=20)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r'$\Theta / \pi$',fontsize = 25)
#[$(h^{-1}{\rm Mpc})^3$]
plt.ylabel(r"$B(k_1,k_2,\Theta)$ [$(h^{-1}{\rm Mpc})^6$]",fontsize=20)
plt.legend(fontsize=18)

plt.xticks(fontsize=15)#, rotation=90)
plt.yticks(fontsize=15)#, rotation=90)
plt.tight_layout()

plt.savefig('../bin/bispectra.png')


# PDF

test_cube = np.load('../dat/processed/test_cube_target.npy')
pred_cube = np.load('../dat/processed/test_cube_final_prediction.npy')
benchmark_cube = np.load('../dat/processed/benchmark_cube.npy')


pred_cube = pred_cube[pred_cube != 0] 
test_cube = test_cube[test_cube != 0] 
benchmark_cube = benchmark_cube[benchmark_cube !=0]

plt.figure()
plt.hist(pred_cube.flatten(),color='crimson' ,label = 'dm2gal',bins=np.logspace(np.log10(1e8),np.log10(1e12), 100),alpha=.5,linestyle='dotted')
#linestyle=('solid','dashed'))
#plt.hist(np.log10(pred_cube.flatten()),bins = 30,histtype='step',ls='solid')
plt.hist(benchmark_cube.flatten(),color='limegreen'  ,label='HOD',bins=np.logspace(np.log10(1e8),np.log10(1e12), 100),alpha=.5,linestyle='dotted')
plt.hist(test_cube.flatten() ,color='slateblue',label='Target',bins=np.logspace(np.log10(1e8),np.log10(1e12), 100),alpha=.5,linestyle='dotted')


plt.xlabel(r"Stellar Mass [${h^{-1}}\rm {M_{\odot}}]$",fontsize = 20)
plt.ylabel("# of voxels",fontsize = 20)
plt.legend(fontsize=15)
plt.xticks(fontsize=12)#, rotation=90)
plt.yticks(fontsize=12)#, rotation=90)

plt.yscale('log')
plt.xscale('log')
plt.xlim([9e7, 2e11])

plt.savefig('../bin/PDF.png')
