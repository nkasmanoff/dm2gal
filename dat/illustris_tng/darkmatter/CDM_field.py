"""

Computes the full dark matter mass density field from the n-body simulation.

"""


#import dependencies
import numpy as np
import MAS_library as MASL
import sys,os,h5py,time
import units_library as UL

dims = 2048
MAS  = 'CIC' 

delta_m = np.zeros((dims,dims,dims), dtype=np.float32)

U = UL.units();  rho_crit = U.rho_crit #h^2 Msun/Mpc^3

# read header
FILE_TYPE = 'darkmatter/'
root = '/scratch/nsk367/pytorch-use/research/dm2gal/dat/illustris_tng/' + FILE_TYPE + 'snapshot/'
# /scratch/gpfs/nk11/dm2gal/dat/illustris-data/hydrosim

prefix_out = 'darkmatter_TNG100-1'

f = h5py.File(root+os.listdir(root)[0],'r')

redshift = f['Header'].attrs[u'Redshift']
Omega_m  = f['Header'].attrs[u'Omega0']
BoxSize  = f['Header'].attrs[u'BoxSize']/1e3  #Mpc/h
Omega_L  = f['Header'].attrs[u'OmegaLambda']
h        = f['Header'].attrs[u'HubbleParam']
#Nall     = f['Header'].attrs[u'Nsubgroups_Total']
filenum  = f['Header'].attrs[u'NumFilesPerSnapshot']
h        = f['Header'].attrs[u'HubbleParam']
Masses   = f['Header'].attrs[u'MassTable']*1e10  #Msun/h
#print('Working with %s at redshift %.0f'%(run,redshift))

f_out = '%s_z=%.1f.hdf5'%(prefix_out,round(redshift))

f.close()

# do a loop over all subfiles in a given snapshot
M_total, start = 0.0, time.time()
for snapshot_tab in os.listdir(root):
    if snapshot_tab != 'wget-log':
        f = h5py.File(root+snapshot_tab,'r')
        #so all header data is the same. 
              
                
        ### CDM ###
        pos  = (f['PartType1/Coordinates'][:]/1e3).astype(np.float32)
        #Masses = f['Mass']
        mass = np.ones(pos.shape[0], dtype=np.float32)*Masses[1] #Msun/h
        MASL.MA(pos, delta_m, BoxSize, MAS, mass)  #CDM
        M_total += np.sum(mass, dtype=np.float64)            
           
                
        f.close()

f = h5py.File(f_out,'w')
f.create_dataset('delta_cdm',  data=delta_m)
f.close()
