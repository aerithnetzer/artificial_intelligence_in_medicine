#!/bin/bash
#SBATCH -A p32234              # Allocation
#SBATCH -p gengpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:10:00             # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --mem=16G               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=1     # Number of Cores (Processors)
#SBATCH --mail-user=aerith.netzer@northwestern.edu
uv add --index-url=https://pypi.nvidia.com "cugraph-cu12"
module load cuda/12.6.2-gcc-12.4.0
NX_CUGRAPH_AUTOCONFIG=True
uv run artificial_intelligence_in_medicine/modeling/graphs.py
