# sampling 


Using the density fields, iterate through and create input-target pairs. These pairs are generated based on a set of pre-defined conditions; each of the 2048^3 points in the hydrodynamic (target) simulation is considered separately. If one of these points contains a value corresponding to > 10^8 solar masses, we save that value, along with the surrounding 1.2 Mpc/h on each side from the input n-body density fields. This surrounding box serves as the context from the dark matter structure, and the network uses it to reconstruct the galaxy / stellar mass near the center of it. 



To run, 

	# python make_samples.py


