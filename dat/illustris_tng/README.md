# illustris-TNG

This repo contains the scripts for downloading density fields from the IllustrisTNG. 

3 density fields used here, the dark matter only simulation, dark matter only fof simulation, and hydrodynamic fof simulation. 

Fof simulations correspond to the friends-of-friends results, and basically means these files only contain the big clumps aggregated into haos and subhalos. 


The steps to collecting the corresponding data types are below. 

Note that all of these scripts assume you have the path to the downloaded illustris files already saved. I have them in the path:

root = 'PATH/dm2gal/illustris-data/SIM_TYPE/sim_files'

To see how you obtain these items, check in the corresponding directories listed below and once you download these files, please run the following python commands. 

## dark_matter

 ### python CDM_field.py


## subhalos

 ### python subhalos.py


## hydro

 ### python hydrosim.py


 The output of these dumps an hdf5 file containing the 2048^3 version of that density field, which we next split into input-output pairs that the model works with in  dm2gal/dat/sampling.