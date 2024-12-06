import os
from datetime import datetime

# Path to save the shell scripts
script_dir = "/t3home/ewiegers/shell_files" # Change this to your desired directory

# Ensure the directory exists
os.makedirs(script_dir, exist_ok=True)

# Define the base script template with a fixed epoch value
base_script = """#!/bin/bash

#SBATCH --job-name=Training_loop
#SBATCH --partition=long
#SBATCH --account=t3
#SBATCH --nodes=1                           # request to run job on single node                                       
#SBATCH --ntasks=5                          # request 8 CPUs
#SBATCH --mem=24000M                        # memory (per job)
#SBATCH --time=00-24:00                     # time in format DD-HH:MM
#SBATCH -o /t3home/ewiegers/logs/%x-%j.out  # Specify the output log file location
#SBATCH -e /t3home/ewiegers/logs/%x-%j.err  # Specify the error log file location

# Define variables
epoch=1                                    # Fix number of epochs
batch_size=256                              #Fix batch size for training
learning_rate=0.001                         # Fix learning rate
nlund=5                                     # Fix number of Lund variables
architecture="{architecture}"               # Architecture must be fixed below
output_dir="/t3home/ewiegers/SVJ_GNN/Plots" # Output directory
alpha=0                                     # Weight of the feature loss during training
test_alpha=0                                # Weight of the feature loss during testing


# Create scratch directory
mkdir -p /scratch/$USER/${{SLURM_JOB_ID}}
export TMPDIR=/scratch/$USER/${{SLURM_JOB_ID}}

# Execute Python script with the variables passed as arguments
python training.py --epoch $epoch \\
                   --batch_size $batch_size \\
                   --learning_rate $learning_rate \\
                   --nlund $nlund \\
                   --architecture $architecture \\
                   --output_dir $output_dir \\
                   --alpha $alpha \\
                   --test_alpha $test_alpha \\

# Clean up scratch directory
rm -rf /scratch/$USER/${{SLURM_JOB_ID}}
"""

# List of architectures (note that the format must be a string with spaces between the layer sizes)
architectures = ["512 256 128 64 32 32"]


# Get current date and time for file naming (without seconds)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Generate scripts for all different architectures
for architecture in architectures:
    # Create script filename using the architecture, epsilon, date, and time (without seconds)
    arch_str = "_".join(architecture.split())  # Convert architecture to a filename-friendly format
    script_filename = os.path.join(script_dir, f"GAE_training_arch_{arch_str}_{current_time}.sh")
    
    # Fill the base script with the current architecture and epsilon
    script_content = base_script.format(architecture=architecture)
    
    # Write the script to the specified directory
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)

    print(f"Created {script_filename}")
    
    # Submit the script using sbatch
    #os.system(f"sbatch {script_filename}")

    # Run the script on the PSI cluster (particularly useful for testing purposes)
    os.system(f"bash {script_filename}")




