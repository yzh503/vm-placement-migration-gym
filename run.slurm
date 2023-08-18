#!/bin/bash

# Partition for the job:
#SBATCH --partition=physical

# The name of the job:
#SBATCH --job-name="rl_training"

# The project ID which this job should run under:
#SBATCH --account="punim0487"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1

# Slurm QoS:
#SBATCH --qos=normal

# Use this email address:
#SBATCH --mail-user=yangzy3@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=2-00:00:00

#SBATCH --output=training-1.log   # Standard output and error log

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load anaconda3/2021.11

# The job command(s):
eval "$(conda shell.bash hook)"
conda activate vm
srun python main.py -e -c config/100.yml -r wr -a ppo -w weights/ppo-wr.pt -l tensorboard -j ppo-wr &
srun python main.py -e -c config/100.yml -r ut -a ppo -w weights/ppo-ut.pt -l tensorboard -j ppo-ut &
srun python main.py -e -c config/100.yml -r kl -a ppo -w weights/ppo-kl.pt -l tensorboard -j ppo-kl &
srun python main.py -e -c config/100.yml -r wr -a caviglione -w weights/caviglione-wr.pt -l tensorboard -j caviglione-wr & 
srun python main.py -e -c config/100.yml -r ut -a caviglione -w weights/caviglione-ut.pt -l tensorboard -j caviglione-ut &
srun python main.py -e -c config/100.yml -r kl -a caviglione -w weights/caviglione-kl.pt -l tensorboard -j caviglione-kl &

wait 

srun python exp_reward.py
srun python exp_migration_ratio.py
srun python exp_beta.py 
srun python exp_performance.py 
srun python exp_suspension.py 
srun python exp_vm_size.py 
srun python exp_var.py 