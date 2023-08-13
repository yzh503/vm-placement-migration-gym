#!/bin/bash

# Partition for the job:
#SBATCH --partition=physical

# The name of the job:
#SBATCH --job-name="rl_experiment"

# The project ID which this job should run under:
#SBATCH --account="punim0487"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=24
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
#SBATCH --time=1-00:00:00

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
srun python exp_migration_ratio.py