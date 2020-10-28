# src


After creating and cleaning the data through the various functions in ../dat, we can now begin to train models that predict the stellar mass distribution throughout the simulation. 

If you want to use the pre-trained best model, download it from the link given, update your path accordingly in test_cube.py (line 29), and then run 

	python test_cube.py
	python hod.py
	python create_plots.py

And this will make the model's reconstructed density field + baseline, and the cosmologically relevant plots. 


If you want to train a model from scratch, 

you can run 

	python submit_jobs.py --gpus 1 --on_cluster

with what ever choice of hyperparameters you like. Note the first time running this script will also compute the class weights for upsampling the higher-mass galaxies during training. After this first time, assuming you don't tweak the # of bins for weighted sampling, this is the only time you'll need to wait for this little extra time :-). 