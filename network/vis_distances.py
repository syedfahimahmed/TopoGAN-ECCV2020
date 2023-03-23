import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file_path_wnet200 = 'C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/TopoSegNetSimple/New_Output/WNet_200_epochs_small/distances.txt'
data_wnet200 = pd.read_csv(file_path_wnet200, delimiter=', ')


file_path_unet120 = 'C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/TopoSegNetSimple/New_Output/Unet_Training_120_epochs_small/distances.txt'
data_unet120 = pd.read_csv(file_path_unet120, delimiter=', ')


file_path_unet50 = 'C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/TopoSegNetSimple/New_Output/Unet_50_epochs_small/distances.txt'
data_unet50 = pd.read_csv(file_path_unet50, delimiter=', ')


file_path_topo50_2 = 'C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/TopoSegNetSimple/New_Output/Topo_50_epochs_small/distances.txt'
data_topo50_2 = pd.read_csv(file_path_topo50_2, delimiter=', ')


file_path_topo50_5 = 'C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/TopoSegNetSimple/New_Output/Topo_50_epochs_small_5/distances.txt'
data_topo50_5 = pd.read_csv(file_path_topo50_5, delimiter=', ')


# Figure 1: Epoch vs Wasserstein Distance 0
plt.figure()
plt.plot(data_wnet200['epoch'], data_wnet200['wasserstein_distance_0'], label=' Wnet-200')
plt.plot(data_unet120['epoch'], data_unet120 ['wasserstein_distance_0'], label=' Unet-120')
plt.plot(data_unet50['epoch'], data_unet50 ['wasserstein_distance_0'], label=' Unet-50')
plt.plot(data_topo50_2['epoch'], data_topo50_2 ['wasserstein_distance_0'], label=' Topo-50 weight:0.0002')
plt.plot(data_topo50_5['epoch'], data_topo50_5 ['wasserstein_distance_0'], label=' Topo-50 weight:0.0005')
plt.xlabel('Epoch')
plt.ylabel('Wasserstein Distance 0 Persistant Homology')
plt.title('Epoch vs Wasserstein Distance 0')
plt.legend()
plt.savefig('wassertein0.png')

# Figure 2: Epoch vs Wasserstein Distance 1
plt.figure()
plt.plot(data_wnet200['epoch'], data_wnet200['wasserstein_distance_1'], label=' Wnet-200')
plt.plot(data_unet120['epoch'], data_unet120 ['wasserstein_distance_1'], label=' Unet-120')
plt.plot(data_unet50['epoch'], data_unet50 ['wasserstein_distance_1'], label=' Unet-50')
plt.plot(data_topo50_2['epoch'], data_topo50_2 ['wasserstein_distance_1'], label=' Topo-50 weight:0.0002')
plt.plot(data_topo50_5['epoch'], data_topo50_5 ['wasserstein_distance_1'], label=' Topo-50 weight:0.0005')
plt.xlabel('Epoch')
plt.ylabel('Wasserstein Distance 1 Persistant Homology')
plt.title('Epoch vs Wasserstein Distance 1')
plt.legend()
plt.savefig('wassertein1.png')

# Figure 3: Epoch vs Bottleneck Distance 0
plt.figure()
plt.plot(data_wnet200['epoch'], data_wnet200['bottleneck_distance_0'], label=' Wnet-200')
plt.plot(data_unet120['epoch'], data_unet120 ['bottleneck_distance_0'], label=' Unet-120')
plt.plot(data_unet50['epoch'], data_unet50 ['bottleneck_distance_0'], label=' Unet-50')
plt.plot(data_topo50_2['epoch'], data_topo50_2 ['bottleneck_distance_0'], label=' Topo-50 weight:0.0002')
plt.plot(data_topo50_5['epoch'], data_topo50_5 ['bottleneck_distance_0'], label=' Topo-50 weight:0.0005')
plt.xlabel('Epoch')
plt.ylabel('Bottleneck Distance 0 Persistant Homology')
plt.title('Epoch vs Bottleneck Distance 0')
plt.legend()
plt.savefig('bottleneck0.png')

# Figure 4: Epoch vs Bottleneck Distance 1
plt.figure()
plt.plot(data_wnet200['epoch'], data_wnet200['bottleneck_distance_1'], label=' Wnet-200')
plt.plot(data_unet120['epoch'], data_unet120 ['bottleneck_distance_1'], label=' Unet-120')
plt.plot(data_unet50['epoch'], data_unet50 ['bottleneck_distance_1'], label=' Unet-50')
plt.plot(data_topo50_2['epoch'], data_topo50_2 ['bottleneck_distance_1'], label=' Topo-50 weight:0.0002')
plt.plot(data_topo50_5['epoch'], data_topo50_5 ['bottleneck_distance_1'], label=' Topo-50 weight:0.0005')
plt.xlabel('Epoch')
plt.ylabel('Bottleneck Distance 1 Persistant Homology')
plt.title('Epoch vs Bottleneck Distance 1')
plt.legend()
plt.savefig('bottleneck1.png')

# Show the figures
plt.show()