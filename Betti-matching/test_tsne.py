import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
import os
import numpy as np
import random
import model
from sklearn.cluster import KMeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = model.AutoEncoder(64,1,256,1,16,'unet').to(device)

# Load pre-trained model
model_path = "C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Demo_Betti/Dream3D_dice_betti1/models/dream3d/DiceBettiMatching_superlevel_relative_True_alpha_0.5_scratch/best_model_dict.pth"  # replace with your model's path and name
checkpoint = torch.load(model_path)

unet_model.load_state_dict(checkpoint['model'])
unet_model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Get image file paths
'''input_data_path = "C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Demo_Betti/Betti-matching/data/dream3d/images"
all_image_files = [os.path.join(input_data_path, file) for file in os.listdir(input_data_path)]
random.seed(0)  # ensuring reproducibility
random.shuffle(all_image_files)  # randomize the order of the files

# Select a random subset of 2000 images
input_image_files = all_image_files[:2000]'''

# Get all image file paths
input_data_path = "C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Demo_Betti/Betti-matching/data/dream3d/images"
all_image_files = [os.path.join(input_data_path, file) for file in os.listdir(input_data_path)]

# Shuffle the list of image files
random.seed(0)  # Ensuring reproducibility

# Select a random subset of 2000 images with 400 images from each consecutive 2000 image block
input_image_files = []
start_index = 0
for _ in range(5):
    subset = all_image_files[start_index:start_index+2000]
    random_subset = random.sample(subset, 400)
    input_image_files.extend(random_subset)
    start_index += 2000
    


# Extract activation maps
activation_maps = []
with torch.no_grad():
    for image_file in input_image_files:
        image = Image.open(image_file)
        input_data = transform(image).unsqueeze(0).to(device)
        output, activations = unet_model(input_data)
        print(activations.shape)
        activation_maps.append(activations.cpu().numpy())
        
activation_maps = np.concatenate(activation_maps)

# Apply t-SNE
tsne = TSNE(n_components=2)
reduced_latent_space_tsne = tsne.fit_transform(activation_maps.reshape(len(activation_maps), -1))

# Combine the x and y coordinates into a feature matrix
features = np.column_stack((reduced_latent_space_tsne[:, 0], reduced_latent_space_tsne[:, 1]))

# Perform K-means clustering
k = 5  # Number of clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(features)

# Get the cluster labels assigned to each data point
labels = kmeans.labels_

# Define a color map for the clusters
colors = ['red', 'blue', 'green', 'purple', 'orange']

# Visualize the clusters on the scatter plot
fig, ax = plt.subplots()
plt.scatter(reduced_latent_space_tsne[:, 0], reduced_latent_space_tsne[:, 1], c=[colors[label] for label in labels], picker=True)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='black')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of Reduced Latent Space')

images = [Image.open(fname) for fname in input_image_files]

def onpick(event):
    ind = event.ind[0]
    fig, ax = plt.subplots()
    ax.imshow(images[ind])
    plt.show()

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()