import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Define the path to your embeddings file
embeddings_path = 'embeddings_clip_RN50_clipdissect_20k.pth'

# Load the embeddings onto the CPU
embeddings_data = torch.load(embeddings_path, map_location=torch.device('cpu'))

# Check if the embeddings_data is a tensor
if isinstance(embeddings_data, torch.Tensor):
    embeddings = embeddings_data.numpy()  # Convert to numpy array for easier handling
else:
    raise ValueError("The embeddings data is not a tensor.")

# Option 1: Load words from a vocabulary file (replace with your actual file)
vocab_file = 'clipdissect_20k.txt'
with open(vocab_file, 'r') as f:
     words_list = [line.strip() for line in f.readlines()]

# Ensure words_list and embeddings match in size
if len(words_list) != embeddings.shape[0]:
    raise ValueError("The number of words does not match the number of embeddings.")

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=3, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)

# Take a subset of the embeddings to visualize (e.g., first 500 words)
subset_size = 50
reduced_subset = reduced_embeddings[1000:1050]
words_subset = words_list[1000:1050]

# Create a DataFrame for easy visualization
df = pd.DataFrame(reduced_subset, columns=['x', 'y', 'z'])
df['word'] = words_subset

# 3D Plot the embeddings with labels
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['x'], df['y'], df['z'], s=10, c='blue')

# Annotate the points with the words
for i, row in df.iterrows():
    ax.text(row['x'], row['y'], row['z'], row['word'], fontsize=8)

# Customize the plot
plt.title("t-SNE 3D visualization of CLIP Text Embeddings", fontsize=16)
ax.set_xlabel('t-SNE Component 1', fontsize=12)
ax.set_ylabel('t-SNE Component 2', fontsize=12)
ax.set_zlabel('t-SNE Component 3', fontsize=12)

plt.savefig('viz_3d.png')
plt.show()
