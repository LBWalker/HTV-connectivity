#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=64
#SBATCH --job-name=particletracking_part10_accuratedepths_modifiedreleasecheck
#SBATCH --output=output_reordered_num_depth_3d_multipl10_multidepth_maxdepth_100minterval_150d_75r_accuratedepths_repeat_printdepths_modifyreleasedepthscheck.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lilyblanche-walker@oist.jp

module purge

#module load python/3.10.2 #original, but may not contain parcels
module load python/3.11
module load ruse/2.0

#python3 main_3D_multiplep_10day.py
#python3 main_3D_multiplep_64day_30r_multi.py
#python3 2d_multip_multidepth_500m_test.py

ruse --stdout python3 reordered_num_depth_3d_multipl10_multidepth_maxdepth_100minterval_150d_75r_accuratedepths_repeat_printdepths_modifyreleasedepthscheck.py
