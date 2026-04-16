#!/bin/bash
#SBATCH -A p32234                          # Allocation
#SBATCH -p normal                          # Partition (normal for longer jobs)
#SBATCH -t 08:00:00                        # Walltime
#SBATCH -N 1                               # Number of Nodes
#SBATCH --mem=128G                         # Memory (loading 3 large graphs + 3 feature JSONs)
#SBATCH --ntasks-per-node=8                # Number of Cores
#SBATCH --job-name=aim_viz_all             # Job name
#SBATCH --output=slurm_%j.out             # Stdout log
#SBATCH --error=slurm_%j.err              # Stderr log
#SBATCH --mail-user=aerith.netzer@northwestern.edu
#SBATCH --mail-type=END,FAIL              # Email on completion or failure

export PATH="$HOME/.local/bin:$PATH"

cd /projects/p32234/projects/aerith/artificial_intelligence_in_medicine

# Sync dependencies
uv sync

# Run the full visualization pipeline
uv run python -m artificial_intelligence_in_medicine.generate_all all
