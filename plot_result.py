# Visualizes GNN training metrics and generates 2D PCA/t-SNE plots of user/post embeddings with interaction type analysis

import json
import matplotlib.pyplot as plt

# ============================================================================
# SECTION 1: Plot Training Metrics from JSON
# ============================================================================

# Load training metrics from JSON file
with open('training_metrics.json', 'r') as f:
    metrics = json.load(f)

# Create a figure with two subplots side by side
plt.figure(figsize=(12, 4))

# Subplot 1: Training loss over epochs
plt.subplot(1, 2, 1)
plt.plot(metrics['epochs'], metrics['train_losses'], 'b-', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)

# Subplot 2: Validation metrics (Recall and NDCG) over epochs
plt.subplot(1, 2, 2)
plt.plot(metrics['epochs'], metrics['val_recalls'], 'g-', label='Recall@10')
plt.plot(metrics['epochs'], metrics['val_ndcgs'], 'r-', label='NDCG@10')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Evaluation Metrics')
plt.legend()
plt.grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# SECTION 2: Visualize User Embeddings using PCA
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import torch

# Load trained user embeddings from checkpoint
checkpoint = torch.load("higgs_embeddings_trained.pt")
user_emb = checkpoint['user_emb'].numpy()  # Convert to numpy array
user_to_idx = checkpoint['user_to_idx']  # User ID to index mapping

# Apply PCA to reduce user embeddings to 2D for visualization
pca = PCA(n_components=2, random_state=42)
user_emb_2d = pca.fit_transform(user_emb)

# Alternative: Use t-SNE for better clustering visualization (commented out, slower)
# tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# user_emb_2d = tsne.fit_transform(user_emb)

# Create scatter plot of 2D user embeddings
plt.figure(figsize=(12, 8))
plt.scatter(user_emb_2d[:, 0], user_emb_2d[:, 1], alpha=0.6, s=50)
plt.title('User Embeddings (2D PCA)', fontsize=16, fontweight='bold')
plt.xlabel('PC1')  # First principal component
plt.ylabel('PC2')  # Second principal component
plt.grid(True, alpha=0.3)
plt.savefig('user_embeddings_pca.png', dpi=300, bbox_inches='tight')
plt.show()

# Print how much variance is explained by the two principal components
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")


# ============================================================================
# SECTION 3: Visualize Post Embeddings Colored by Interaction Type
# ============================================================================

# Load processed data with interaction information
data = torch.load("synthetic_processed_with_semantics.pt")
activity_sub = data['activity_sub']
post_emb = checkpoint['post_emb'].numpy()  # You'll need to save this during training

# Extract interaction types for each post (QT=Quote Tweet, RE=Reply, POST=Original Post)
interaction_types = activity_sub['interaction'].values
color_map = {'QT': 'red', 'RE': 'blue', 'POST': 'green'}
colors = [color_map[inter] for inter in interaction_types]

# Apply PCA to reduce post embeddings to 2D
pca_post = PCA(n_components=2, random_state=42)
post_emb_2d = pca_post.fit_transform(post_emb)

# Plot post embeddings with different colors for each interaction type
plt.figure(figsize=(12, 8))
for inter_type, color in color_map.items():
    mask = interaction_types == inter_type  # Filter posts by interaction type
    plt.scatter(post_emb_2d[mask, 0], post_emb_2d[mask, 1], 
               c=color, label=inter_type, alpha=0.6, s=50)

plt.title('Post Embeddings by Interaction Type (2D PCA)', fontsize=16, fontweight='bold')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('post_embeddings_by_type.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# SECTION 4: Visualize Combined User and Post Embeddings
# ============================================================================

# Stack user and post embeddings vertically to create a combined embedding matrix
all_embeddings = np.vstack([user_emb, post_emb])
# Create labels to track which embeddings are users vs posts
node_types = ['user'] * len(user_emb) + ['post'] * len(post_emb)

# Apply PCA to reduce combined embeddings to 2D
pca_all = PCA(n_components=2, random_state=42)
all_emb_2d = pca_all.fit_transform(all_embeddings)

# Create masks to separate users and posts for different colors
plt.figure(figsize=(14, 10))
user_mask = np.array(node_types) == 'user'
post_mask = np.array(node_types) == 'post'

# Plot users in blue and posts in red
plt.scatter(all_emb_2d[user_mask, 0], all_emb_2d[user_mask, 1], 
           c='blue', label='Users', alpha=0.6, s=40)
plt.scatter(all_emb_2d[post_mask, 0], all_emb_2d[post_mask, 1], 
           c='red', label='Posts', alpha=0.6, s=20)

plt.title('Combined User and Post Embeddings (2D PCA)', fontsize=16, fontweight='bold')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('combined_embeddings.png', dpi=300, bbox_inches='tight')
plt.show()
