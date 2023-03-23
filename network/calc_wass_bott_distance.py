import os
import numpy as np
import gudhi as gd
import gudhi.wasserstein as wasserstein

def get_folders_starting_with(directory, prefix):
    folders = []
    for entry in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, entry)) and entry.startswith(prefix):
            folders.append(os.path.join(directory, entry))
    return folders

def calculate_persistence_distances(data, max_edge_length, max_dimension):
    rc = gd.RipsComplex(points=data, max_edge_length=max_edge_length)
    st = rc.create_simplex_tree(max_dimension=max_dimension)
    persistence = st.persistence()

    # Separate 0 and 1-dimensional persistence pairs
    persistence_0 = st.persistence_intervals_in_dimension(0)
    persistence_1 = st.persistence_intervals_in_dimension(1)

    return persistence_0, persistence_1

def save_distances_to_file(file_path, epoch_distances):
    with open(file_path, "w") as f:
        f.write("epoch, wasserstein_distance_0, wasserstein_distance_1, bottleneck_distance_0, bottleneck_distance_1\n")
        for epoch, distances in epoch_distances.items():
            f.write(f"{epoch}, {distances[0]}, {distances[1]}, {distances[2]}, {distances[3]}\n")
    print(f"Distances saved to {file_path}")

# Parameters
directory_path = "C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/TopoSegNetSimple/New_Output/Unet_Training_120_epochs_small"  # Replace with the path to your directory
folder_prefix = 'epoch'
file_name = 'output_00001.npy'
target_numpy_file = "C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/TopoSegNetSimple/datagenerator/CREMI_AB_64_64_test0/boundaries/boundary_00001.npy"  # Replace with the path to the target numpy file
max_edge_length = 90.6
max_dimension = 2
output_file = 'distances1.txt'

# Load the target numpy file for comparison
target_data = np.load(target_numpy_file)
target_np = np.column_stack(np.where(target_data == 1))
target_persistence_0, target_persistence_1 = calculate_persistence_distances(target_np, max_edge_length, max_dimension)

folders = get_folders_starting_with(directory_path, folder_prefix)
epoch_distances = {}

for folder in folders:
    file_path = os.path.join(folder, file_name)
    # replace the backslashes with forward slashes
    file_path = file_path.replace("\\", "/")
    if os.path.isfile(file_path):
        epoch = int(folder.split('_')[-1])  # Extract epoch number from folder name
        if (epoch==0) or ((epoch+1) % 5 == 0):
            print(f"Processing epoch {epoch}...")
            data = np.load(file_path)
            output_np = np.column_stack(np.where(data > 0.3))
            output_persistence_0, output_persistence_1 = calculate_persistence_distances(output_np, max_edge_length, max_dimension)
            wasserstein_distance_0 = wasserstein.wasserstein_distance(target_persistence_0, output_persistence_0, order=2, internal_p=2.0)
            wasserstein_distance_1 = wasserstein.wasserstein_distance(target_persistence_1, output_persistence_1, order=2, internal_p=2.0)
            bottleneck_distance_0 = gd.bottleneck_distance(target_persistence_0, output_persistence_0, e=0)
            bottleneck_distance_1 = gd.bottleneck_distance(target_persistence_1, output_persistence_1, e=0)
            epoch_distances[epoch] = (wasserstein_distance_0, wasserstein_distance_1, bottleneck_distance_0, bottleneck_distance_1)
            print(f"Epoch {epoch}: {epoch_distances[epoch]}")
    else:
        print(f"File '{file_name}' not found in folder '{folder}'")

output_file_path = os.path.join(directory_path, output_file)
save_distances_to_file(output_file_path, epoch_distances)