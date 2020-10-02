"""
CDM field.py


This script computes a mass density field, using all particle information from the nbody simulation specified below. 

"""

import numpy as np
import MAS_library as MASL
import sys,os,h5py,time
import units_library as UL
import HI_library as HIL

U = UL.units();  rho_crit = U.rho_crit #h^2 Msun/Mpc^3
################################ INPUT ########################################
snapnums = np.array([99])

runs  = ['/simons/scratch/sgenel/Illustris_IllustrisTNG_public_data_release/L75n1820TNG_DM']
fouts = ['TNG100-1DM']

dims = 2048

MAS = 'CIC'
##############################################################################

for fout,run in zip(fouts,runs):

    # do a loop over the different redshifts
    for snapnum in snapnums:

        # define the array hosting delta_HI and delta_m
        delta_m  = np.zeros((dims,dims,dims), dtype=np.float32)

        # read header
        snapshot = '%s/output/snapdir_%03d/snap_%03d'%(run,snapnum, snapnum)
        f = h5py.File(snapshot+'.0.hdf5', 'r')
        redshift = f['Header'].attrs[u'Redshift']
        BoxSize  = f['Header'].attrs[u'BoxSize']/1e3  #Mpc/h
        filenum  = f['Header'].attrs[u'NumFilesPerSnapshot']
        Omega_m  = f['Header'].attrs[u'Omega0']
        Omega_L  = f['Header'].attrs[u'OmegaLambda']
        h        = f['Header'].attrs[u'HubbleParam']
        Masses   = f['Header'].attrs[u'MassTable']*1e10  #Msun/h
        f.close()

        print 'Working with %s at redshift %.0f'%(run,redshift)
        f_out = '%s_z=%.1f.hdf5'%(fout,round(redshift))

        # if file exists move on
        if os.path.exists(f_out):  continue

        # do a loop over all subfiles in a given snapshot
        M_total, start = 0.0, time.time()
        for i in xrange(filenum):

            snapshot = '%s/output/snapdir_%03d/snap_%03d.%d.hdf5'\
                       %(run,snapnum,snapnum,i)
            f = h5py.File(snapshot, 'r')

            ### CDM ###
            pos  = (f['PartType1/Coordinates'][:]/1e3).astype(np.float32)        
            mass = np.ones(pos.shape[0], dtype=np.float32)*Masses[1] #Msun/h
            MASL.MA(pos, delta_m, BoxSize, MAS, mass)  #CDM
            M_total += np.sum(mass, dtype=np.float64)

            f.close()

            print '%03d -----> Omega_cdm = %.4f  : %6.0f s'\
                %(i, M_total/(BoxSize**3*rho_crit), time.time()-start)

        f = h5py.File(f_out,'w')
        f.create_dataset('delta_cdm',  data=delta_m)
        f.close()