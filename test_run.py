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
test_dataset = ["BraTS20", "BraTS21", "BraTS23", "MSD"]
# Define your commands
commands = [
    [
        "python", "Comparision_test.py", "base.txt",
        "--dataset_name", dataset[0],
        "--device", "cuda:0",
        "--batch_size", "1",
        "--seed", "True",
        "--folder_name", "Write Folder name in which checkpoints have been saved",
        "--test_dataset", test_dataset[0]
    ]
]


# Execute each command sequentially
for command in commands:
    run_command(command)
    free_memory()  # Free memory after each command
