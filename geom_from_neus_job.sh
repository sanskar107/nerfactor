#!/bin/bash
#SBATCH --job-name=transform
#SBATCH -p cpu
#SBATCH --qos=normal
#SBATCH -c 8
#SBATCH --output="geom_neus.log"
#SBATCH --open-mode=append

python geom_from_neus.py