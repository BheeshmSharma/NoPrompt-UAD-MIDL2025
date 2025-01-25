import subprocess
import torch
import gc

def run_command(command):
    """Run a shell command and handle errors."""
    try:
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
        print("Command completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {' '.join(command)}")
        print(f"Error: {e}")
        exit(1)  # Exit if any command fails

def free_memory():
    """Free GPU and CPU memory."""
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleanup completed.")

dataset = ["BraTS20", "BraTS21", "BraTS23", "MSD"]
# Define your commands
commands = [
    [
        "python", "train_test.py", "base.txt",
        "--dataset_name", dataset,
        "--fold", "[0,1]",
        "--device", "cuda",
        "--learning_rate", "1e-2",
        "--batch_size", "16",
        "--seed", "True",
        "--num_epochs", "200",
        "--folder_name", "try_ActivePointEmbeddingLoss_MSE2_0.4_FPpenalty",
        "--pre_training", "False"
    ]
]

# Execute each command sequentially
for command in commands:
    run_command(command)
    free_memory()  # Free memory after each command
