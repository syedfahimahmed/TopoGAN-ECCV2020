import os
import numpy as np
import gudhi as gd
import gudhi.wasserstein as wasserstein
import matplotlib.pyplot as plt

# Define the paths to the two folders
folder1 = "C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/TopoSegNetSimple/datagenerator/CREMI_AB_64_64_test0/boundaries"
folder2 = "C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/TopoSegNetSimple/Output/Topo_Training_ep_300_val/epoch_99"
output_folder = "C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/TopoSegNetSimple/Output/Topo_Training_ep_300_val"

# Read the index file in folder1 to get the list of file names
with open(os.path.join(folder1, 'index.txt'), 'r') as f:
    files1 = [line.strip() for line in f]

# Get the list of files in folder2
files2 = os.listdir(folder2)

# Initialize a list to store the distances
wass_0 = []
wass_1 = []
bott_0 = []
bott_1 = []
step = 0
# Iterate over corresponding pairs of files
for file1, file2 in zip(files1, files2):
    
    # Load the two files as numpy arrays
    data1 = np.load(os.path.join(folder1, file1))
    targets_np = np.transpose(data1)
    
    outputs_np = np.load(os.path.join(folder2, file2))
    
    # compute persistent homology for 0 and 1 homology
    points0_outputs = np.column_stack(np.where(outputs_np > 0.3))
    points0_targets = np.column_stack(np.where(targets_np == 1))
    rips_complex0_outputs = gd.RipsComplex(points=points0_outputs, max_edge_length=90.6)
    rips_complex0_targets = gd.RipsComplex(points=points0_targets, max_edge_length=90.6)
    simplex_tree0_outputs = rips_complex0_outputs.create_simplex_tree(max_dimension = 2)
    simplex_tree0_targets = rips_complex0_targets.create_simplex_tree(max_dimension = 2)
    diag0_outputs = simplex_tree0_outputs.persistence()
    diag0_targets = simplex_tree0_targets.persistence()

    inter_tar0 = simplex_tree0_targets.persistence_intervals_in_dimension(0)
    inter_out0 = simplex_tree0_outputs.persistence_intervals_in_dimension(0)

    inter_tar1 = simplex_tree0_targets.persistence_intervals_in_dimension(1)
    inter_out1 = simplex_tree0_outputs.persistence_intervals_in_dimension(1)

    # Plot the persistent diagram
    #gd.plot_persistence_diagram(diag0_targets)
    #plt.show()

    # Calculate the Wasserstein distance between the diagrams
    wasserstein_distance0 = wasserstein.wasserstein_distance(inter_tar0, inter_out0, order=2, internal_p=2.0)
    wasserstein_distance1 = wasserstein.wasserstein_distance(inter_tar1, inter_out1, order=2, internal_p=2.0)

    #Calculate the Bottleneck distance between the diagrams
    bottleneck_distance0 = gd.bottleneck_distance(inter_tar0, inter_out0, e=0)
    bottleneck_distance1 = gd.bottleneck_distance(inter_tar1, inter_out1, e=0)
    
    # Add the distance to the list
    wass_0.append(wasserstein_distance0)
    wass_1.append(wasserstein_distance1)
    
    bott_0.append(bottleneck_distance0)
    bott_1.append(bottleneck_distance1)
    print("step : ", step)
    step += 1

# Compute the average distance
# Calculate the average distances
avg_wass_0 = np.mean(wass_0)
avg_wass_1 = np.mean(wass_1)
avg_bott_0 = np.mean(bott_0)
avg_bott_1 = np.mean(bott_1)

with open(os.path.join(output_folder, 'epoch_99_distances.txt'), 'w') as f:
    f.write(f'Average Wasserstein distance (0-dimensional): {avg_wass_0}\n')
    f.write(f'Average Wasserstein distance (1-dimensional): {avg_wass_1}\n')
    f.write(f'Average Bottleneck distance (0-dimensional): {avg_bott_0}\n')
    f.write(f'Average Bottleneck distance (1-dimensional): {avg_bott_1}\n')