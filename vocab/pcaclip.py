import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

# Load the embeddings from the .pth file
embeddings_path = 'embeddings_clip_RN50_clipdissect_20k.pth'

# Load the embeddings and words
embeddings_data = torch.load(embeddings_path, map_location=torch.device('cpu'))
print(type(embeddings_data))
print(embeddings_data.shape if isinstance(embeddings_data, torch.Tensor) else "Not a tensor")

# Assuming embeddings_data is a dictionary with words as keys and embeddings as values
words = list(embeddings_data.keys())  # List of words
embeddings = np.array([embeddings_data[word] for word in words])  # Corresponding embeddings

# Reduce dimensionality using t-SNE (you can adjust n_iter for more precision if needed)
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)

# Take a subset of the embeddings to visualize, e.g., the first 500 words (you can adjust as needed)
subset_size = 500
reduced_subset = reduced_embeddings[:subset_size]
words_subset = words[:subset_size]

# Create a DataFrame for visualization
df = pd.DataFrame(reduced_subset, columns=['x', 'y'])
df['word'] = words_subset

# Plot the embeddings with labels
plt.figure(figsize=(10, 8))
plt.scatter(df['x'], df['y'], s=10, c='blue')

# Annotate the points with the words
for i, row in df.iterrows():
    plt.annotate(row['word'], (row['x'], row['y']), fontsize=8)

# Customize the plot
plt.title("t-SNE visualization of CLIP Text Embeddings", fontsize=16)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.show()

