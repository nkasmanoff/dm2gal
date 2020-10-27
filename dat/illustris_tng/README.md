# illustris-TNG

This repo contains the scripts for downloading density fields from the IllustrisTNG. 

3 density fields used here, the dark matter only simulation, dark matter only fof simulation, and hydrodynamic fof simulation. 

Fof simulations correspond to the friends-of-friends results, and basically means these files only contain the big clumps aggregated into haos and subhalos. 


The steps to collecting the corresponding data types are below. 

Note that all of these scrips assume you have the path to the downloaded illustris files already saved somewhere locally. I have them in the path:

root = 'PATH/dm2gal/illustris-data/SIM_TYPE/sim_files'

## dark_matter

 ### python CDM_field.py


## subhalos

 ### python subhalos.py


## hydro

 ### python hydrosim.py


 The output of these scripts will dump an hdf5 file containing the 2048^3 version of that density field, which is partitioned into model ready chunks in dm2gal/dat/sampling.