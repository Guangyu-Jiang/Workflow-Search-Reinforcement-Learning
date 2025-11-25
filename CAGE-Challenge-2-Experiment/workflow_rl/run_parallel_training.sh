#!/bin/bash

# Script to run parallel workflow training

echo "Starting Parallel Workflow Training with 25 environments"
echo "========================================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CAGE2

# Set Python path
export PYTHONPATH=/home/ubuntu/CAGE2/-cyborg-cage-2:$PYTHONPATH

# Run the parallel training
python /home/ubuntu/CAGE2/-cyborg-cage-2/workflow_rl/parallel_train_workflow_rl.py

echo "Training complete!"
