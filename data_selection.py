"""Data selection for SSMPD (paper Sec. III-D).

Encodes every training scene with an ImageNet-pretrained ResNet-50, groups the
scenes with K-means, then samples uniformly by Euclidean distance inside each
cluster, so a small labeled subset still covers diverse scenes. The selected
list is what the supervised teacher is trained on.

Edit `root`, `img_set` and the cluster/sample counts below, then run:
    python data_selection.py
"""
from sklearn.cluster import KMeans
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import numpy as np
import os
import torch
from scipy.spatial import distance

print(f"\nread file\n")

root = 'data/kaist-rgbt/'

# train

img_set = f"train-all-02.txt"

# collect the real image paths
real_img_paths = []

# read the image list
with open("data/kaist-rgbt/imageSets/train-all-02.txt", "r") as f:
    # one image id per line
    for line in f:
        # strip the newline
        img_path = line.strip()

        # split the id into its path components
        path_parts = img_path.split("/")

        # insert "visible" before the file name
        path_parts.insert(-1, "visible")

        # build the path on disk
        real_img_path = os.path.join(f"data/kaist-rgbt/images/{path_parts[0]}/{path_parts[1]}/{path_parts[2]}/{path_parts[3]}.jpg")
        
        # keep it
        real_img_paths.append(real_img_path)

print(f"\ndone.\n")

# transform used before feature extraction
transform = Compose([
    Resize((224, 224)),  # ResNet input size
    ToTensor(),  # to tensor
])

print(f"\nread pretrain weights\n")

# ImageNet-pretrained ResNet-50 as the scene encoder
model = resnet50(pretrained=True)

# adapt the first convolution to grayscale input
#model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# adjust the number of input channels
# the pretrained weights are 3-channel, so average them over the channel axis
#model.conv1.weight.data = model.conv1.weight.data.mean(dim=1, keepdim=True)

model = model.to('cuda')
model.eval()

print(f"\ndone.\n")

# feature vectors
features = []

print(f"\nfeature extract\n")

# encode every image
for i, img_file in enumerate(real_img_paths):
    img = Image.open(img_file)
    img = transform(img).unsqueeze(0)  # add the batch dimension
    img = img.to('cuda')
    
    # forward pass
    with torch.no_grad():
        feature = model(img)
    
    features.append(feature.cpu().numpy())

print(f"\ndone.\n")

# stack the features
features = np.vstack(features)

#num_clusters = [50, 100, 150, 200, 250, 300]
num_clusters = [350,400,450,500,550,600]

for num_cluster in num_clusters:
    # K-means clustering
    print(f"\nstart rgb {num_cluster} clustering\n")
    
    kmeans = KMeans(n_clusters=num_cluster, random_state=42).fit(features)

    print(f"\ndone\n")

    # store the clustering result
    cluster_ids = kmeans.predict(features)

    # store the cluster assignment together with the file name
    np.savez(f"clustering_result_uniform_rgb_{num_cluster}.npz", img_files=real_img_paths, cluster_ids=cluster_ids)

    # Load the clustering result
    data = np.load(f"clustering_result_uniform_rgb_{num_cluster}.npz")
    img_files = data['img_files']
    cluster_ids = data['cluster_ids']

    # Determine the total number of samples to select (10% of the dataset)
    n_total_samples = len(img_files) * 10 // 100
    print(f"\n{num_cluster} total samples : {n_total_samples}\n")
    # If total number of samples is less than the number of clusters, 
    # ensure that at least one sample is selected from each cluster
    #if n_total_samples < 100:
        #n_total_samples = 100


    # Determine the number of samples to select from each cluster
    n_samples_per_cluster_base = n_total_samples // num_cluster
    n_clusters_extra_samples = n_total_samples % num_cluster

    # Initialize a list to hold the selected image file names
    selected_img_files = []

    # For each cluster ID
    for cluster_id in range(kmeans.n_clusters):
        # Get the list of image files in this cluster
        img_files_in_cluster = [img_file for img_file, cluster_id_ in zip(img_files, cluster_ids) if cluster_id_ == cluster_id]
        
        # Get the list of feature vectors in this cluster
        features_in_cluster = [feature for feature, cluster_id_ in zip(features, cluster_ids) if cluster_id_ == cluster_id]

        # Compute the distances from the cluster center to each sample
        cluster_center = kmeans.cluster_centers_[cluster_id]
        distances = [distance.euclidean(feature, cluster_center) for feature in features_in_cluster]

        # Sort the image files in this cluster by distance to the cluster center
        img_files_in_cluster = [img_file for _, img_file in sorted(zip(distances, img_files_in_cluster))]

        # Determine the number of samples to select from this cluster
        n_samples = n_samples_per_cluster_base + (1 if cluster_id < n_clusters_extra_samples else 0)

        # Uniformly select samples from this cluster
        step_size = max(len(img_files_in_cluster) // n_samples, 1)
        #step_size = max(n_total_samples // num_cluster, 1)
        selected_img_files_in_cluster = img_files_in_cluster[::step_size][:n_samples]

        # Add the selected image files to the main list
        selected_img_files.extend(selected_img_files_in_cluster)

    # Save the selected image file names to a new text file
    with open(f"cluster/rgb_10selected_img_files_uniform_rgb_{num_cluster}.txt", "w") as f:
        for img_file in selected_img_files:
            f.write(img_file + "\n")
    print(f"\n{num_cluster} is done.\n")

print("All jobs done.")