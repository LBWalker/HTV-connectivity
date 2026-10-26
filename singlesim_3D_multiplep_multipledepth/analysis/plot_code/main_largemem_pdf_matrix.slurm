#!/bin/bash
#SBATCH --partition=largemem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=750G
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=plots_100part_150d_75r_accuratedepths_repeat
#SBATCH --output=output_pdfplots_pdfmatrix_probmatrix_reordered_num_depth_3d_multipl100_multidepth_maxdepth_100minterval_150d_75r_accuratedepths_repeat_printdepths_modifyreleasedepthscheck.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lily.walker@oist.jp

module purge

#module load python/3.10.2 #original, but may not contain parcels
module load python/3.11
module load ruse/2.0


ruse --stdout python3 pdfplots_pdfmatrix_probmatrix_reordered_num_depth_3d_multipl100_multidepth_maxdepth_100minterval_150d_75r_accuratedepths_repeat_printdepths_modifyreleasedepthscheck.py
