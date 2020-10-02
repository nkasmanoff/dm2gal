#!/bin/bash
##SBATCH --partition gpu4_medium
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 0-08:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --output jupyter-notebook-%J.log 

# Thank you, Yale Center for Research Computing, for this script
# modified for use on BigPurple in NYU Langone Medical Center by Paul Glick
# modified again for use on Tiger in Princeton by Noah Kasmanoff

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
#cluster=$(hostname -f | awk -F"." '{print $2}')
case "$SLURM_SUBMIT_HOST" in
tiger.princeton.edu)
	login_node=tiger.princeton.edu
	;;
	
esac



# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@${login_node} 

For more info and how to connect from windows, 
   see research.computing.yale.edu/jupyter-nb
Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${login_node}
SSH login: $user
SSH port: 22
Use a Browser on your local machine to go to:
http://localhost:${port}  (prefix w/ https:// if using password)
Use the token string from the URL printed below and add it to the URL above
"



# load modules or conda environments here
# e.g. farnam:
# module load Python/2.7.13-foss-2016b 
# module add anaconda2/gpu/5.2.0
#module add anaconda3/cpu/5.2.0 
source activate dm2gal
# DON'T USE ADDRESS BELOW. 
# DO USE TOKEN BELOW
jupyter-notebook --no-browser --port=${port} --ip=${node}
