# hydro

To obtain the data from the TNG which is transformed into a density field, please run the following commands

	cd path/dm2gal/dat/illustrus_tng/hydro/groupcat/groups_099/

	wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 --content-disposition --header="API-Key: fa5bf6656c73e72828d9c1a899841661" "http://www.tng-project.org/api/TNG100-1/files/groupcat-99/?format=api"


Once this is finished loading,

	python hydrosim.py