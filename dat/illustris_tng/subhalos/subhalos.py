"""

Computes the density field for the subhalo only version of the same nbody simulation in the darkmatter folder.


"""

import numpy as np
import MAS_library as MASL
import sys,os,h5py,time
import units_library as UL

dims = 2048
MAS  = 'CIC'
delta = np.zeros((dims,dims,dims), dtype=np.float32)
U = UL.units();  rho_crit = U.rho_crit #h^2 Msun/Mpc^3




# read header
FILE_TYPE = 'subhalos/'
root = '/scratch/nsk367/pytorch-use/research/dm2gal/dat/illustris_tng/' + FILE_TYPE + 'groupcat/groups_099/'
# /scratch/gpfs/nk11/dm2gal/dat/illustris-data/subhalos

prefix_out = 'subhalos_TNG100-1'


f = h5py.File(root+os.listdir(root)[0],'r')
redshift = f['Header'].attrs[u'Redshift']
Omega_m  = f['Header'].attrs[u'Omega0']
BoxSize  = f['Header'].attrs[u'BoxSize']/1e3  #Mpc/h
Omega_L  = f['Header'].attrs[u'OmegaLambda']
h        = f['Header'].attrs[u'HubbleParam']
Nall     = f['Header'].attrs[u'Nsubgroups_Total']

f_out = '%s_z=%.1f.hdf5'%(prefix_out,round(redshift))

do_Group = False
do_Subhalo = True
    # this is always going to be for a hydrodynamic simulation
f.close()
# do a loop over all subfiles in a given snapshot
M_total, start = 0.0, time.time()
for fof_subhalo_tab in os.listdir(root):
    if fof_subhalo_tab != 'wget-log':
        f = h5py.File(root+fof_subhalo_tab,'r')
        #so all header data is the same. 
        #now let's look at file specific attribtues. 
        
        ### CDM ###
        print(fof_subhalo_tab)

            
        N_subhalos = f['Header'].attrs['Nsubgroups_ThisFile']
        if N_subhalos > 0:
            pos = (f['Subhalo/SubhaloPos'][:]/1e3).astype(np.float32)
           # print("Found pos")
        # subhalo_positions.append(subhalo_pos)
            mass = f['Subhalo/SubhaloMass'][:]*1e10  #SubhaloMass
            MASL.MA(pos, delta, BoxSize, MAS, mass)  #stars
            M_total += np.sum(mass, dtype=np.float64)

                     
        f.close()
#print(delta) 

f = h5py.File(f_out,'w')
f.create_dataset('density',  data=delta)
f.close()
