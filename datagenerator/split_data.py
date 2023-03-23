import os
import shutil
import pandas as pd

def split_data(source_dir, train_dir, test_dir, ratio=0.8):
    with open(os.path.join(source_dir, "index.txt"), "r") as index_file:
        files = [line.strip() for line in index_file.readlines()]

    train_size = int(len(files) * ratio)
    train_files = files[:train_size]
    test_files = files[train_size:]
    
    with open(os.path.join(train_dir, "index.txt"), "w") as train_index:
        for f in train_files:
            shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))
            train_index.write(f + "\n")
    
    with open(os.path.join(test_dir, "index.txt"), "w") as test_index:
        for f in test_files:
            shutil.copy(os.path.join(source_dir, f), os.path.join(test_dir, f))
            test_index.write(f + "\n")

dataset_base_path = "C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/dataset"
dataset_path = os.path.join(dataset_base_path, "ssynth10k")

# Create training and testing directories
os.makedirs(os.path.join(dataset_base_path, "training/boundaries"), exist_ok=True)
os.makedirs(os.path.join(dataset_base_path, "training/distances"), exist_ok=True)
os.makedirs(os.path.join(dataset_base_path, "training/images"), exist_ok=True)

os.makedirs(os.path.join(dataset_base_path, "testing/boundaries"), exist_ok=True)
os.makedirs(os.path.join(dataset_base_path, "testing/distances"), exist_ok=True)
os.makedirs(os.path.join(dataset_base_path, "testing/images"), exist_ok=True)

# Split .npy files for each folder
split_data(os.path.join(dataset_path, "boundaries"), os.path.join(dataset_base_path, "training/boundaries"), os.path.join(dataset_base_path, "testing/boundaries"))
split_data(os.path.join(dataset_path, "distances"), os.path.join(dataset_base_path, "training/distances"), os.path.join(dataset_base_path, "testing/distances"))
split_data(os.path.join(dataset_path, "images"), os.path.join(dataset_base_path, "training/images"), os.path.join(dataset_base_path, "testing/images"))

# Split params.csv
params_df = pd.read_csv(os.path.join(dataset_path, "params.csv"))
train_size = int(len(params_df) * 0.8)
train_params_df = params_df.iloc[:train_size]
test_params_df = params_df.iloc[train_size:]

train_params_df.to_csv(os.path.join(dataset_base_path, "training/params.csv"), index=False)
test_params_df.to_csv(os.path.join(dataset_base_path, "testing/params.csv"), index=False)

print("Spliting data into training and testing folders complete!")