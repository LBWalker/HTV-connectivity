#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=64
#SBATCH --job-name=3D_density_plot
#SBATCH --output=density_plot.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lily.walker@oist.jp

module purge

#module load python/3.10.2 #original, but may not contain parcels
module load python/3.11

python3 3D_density_plot_150d_75r.py


