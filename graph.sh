#!/bin/bash
#SBATCH -A p32234              # Allocation
#SBATCH -p short
#SBATCH -t 04:00:00             # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --mem=64G               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=4     # Number of Cores (Processors)
#SBATCH --mail-user=aerith.netzer@northwestern.edu

uv run artificial_intelligence_in_medicine/modeling/first_analysis.py
