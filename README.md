# dm2gal


Maps of cosmic structure produced by galaxy surveys are one of the key tools for answering fundamental questions about the Universe. Accurate theoretical predictions for these quantities are needed to maximize the scientific return of these programs. Hydrodynamic simulations are one of the most powerful techniques to accomplish this; unfortunately, these simulations are very computationally expensive. Alternatively, gravity-only simulations are cheaper, but do not predict the locations and properties of galaxies in the cosmic web. In this work, we use convolutional neural networks to paint galaxy stellar masses on top of the dark matter field generated by gravity-only simulations. Our model outperforms the state-of-the-art benchmark model and allows the generation of fast and accurate models of the observed galaxy distribution. 



## Methodology


We use data from the state-of-the-art magneto-hydrodynamic simulation IllustrisTNG100-1, and its gravity-only counterpart, IllustrisTNG100-1_DM, at present time. Those simulations contain, among other things, the position and mass of all particles in the simulations. Each simulation also contains a catalogue of dark matter halos with their properties (e.g. mass and position). We use the Cloud-in-Cell (CIC) mass assignment interpolation scheme to construct the 3D stellar mass and dark matter mass fields from the particle positions and masses of the hydrodynamic and gravity-only simulations, respectively. Since galaxies are expected to reside in dark matter subhalos, we facilitate the training of the network by using also a 3D field with the mass-weighted subhalo field, that we construct from the gravity-only simulation. The 3D fields span a volume of $(75~h^{-1}{\rm Mpc})^3$ (one ${\rm Mpc}$ corresponds to 3.26 million light-years) and they contain $2048^3$ voxels. 



### 1. Download density field arrays. 
	See dat/illustris-data

### 2. Generate sub-fields.
	See dat/reconstruction-data

### 3. Train model(s) 
	See src/

### 4. Compute cosmological statistics
	See src/


Current methods include visualization of test subcube demonstrating model reconstruction capabilities, summary statistics,train loss curve, power spectrums, predicted vs target mass distribution, and more. 



## Results

Power Spectrum           |  Bispectrum
:-------------------------:|:-------------------------:
![](https://github.com/nkasmanoff/dm2gal/blob/main/bin/power_spectrum.png) |  ![](https://github.com/nkasmanoff/dm2gal/blob/main/bin/bispectra.png)



### Related Work

https://github.com/siyucosmo/ML-Recon

https://github.com/jhtyip/From-Dark-Matter-to-Galaxies-with-Convolutional-Neural-Networks


### Acknowledgments
Too many!
