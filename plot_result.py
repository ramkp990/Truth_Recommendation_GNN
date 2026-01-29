import json
import matplotlib.pyplot as plt

with open('training_metrics.json', 'r') as f:
    metrics = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(metrics['epochs'], metrics['train_losses'], 'b-', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(metrics['epochs'], metrics['val_recalls'], 'g-', label='Recall@10')
plt.plot(metrics['epochs'], metrics['val_ndcgs'], 'r-', label='NDCG@10')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Evaluation Metrics')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import torch

# Load your trained embeddings
checkpoint = torch.load("higgs_embeddings_trained.pt")
user_emb = checkpoint['user_emb'].numpy()
user_to_idx = checkpoint['user_to_idx']

# Reduce to 2D using PCA or t-SNE
pca = PCA(n_components=2, random_state=42)
user_emb_2d = pca.fit_transform(user_emb)

# Or use t-SNE for better clustering (slower)
# tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# user_emb_2d = tsne.fit_transform(user_emb)

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(user_emb_2d[:, 0], user_emb_2d[:, 1], alpha=0.6, s=50)
plt.title('User Embeddings (2D PCA)', fontsize=16, fontweight='bold')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True, alpha=0.3)
plt.savefig('user_embeddings_pca.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Load post embeddings and interaction types
data = torch.load("synthetic_processed_with_semantics.pt")
activity_sub = data['activity_sub']
post_emb = checkpoint['post_emb'].numpy()  # You'll need to save this during training

# Get interaction types for coloring
interaction_types = activity_sub['interaction'].values
color_map = {'QT': 'red', 'RE': 'blue', 'POST': 'green'}
colors = [color_map[inter] for inter in interaction_types]

# Reduce to 2D
pca_post = PCA(n_components=2, random_state=42)
post_emb_2d = pca_post.fit_transform(post_emb)

# Plot with interaction types
plt.figure(figsize=(12, 8))
for inter_type, color in color_map.items():
    mask = interaction_types == inter_type
    plt.scatter(post_emb_2d[mask, 0], post_emb_2d[mask, 1], 
               c=color, label=inter_type, alpha=0.6, s=50)

plt.title('Post Embeddings by Interaction Type (2D PCA)', fontsize=16, fontweight='bold')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('post_embeddings_by_type.png', dpi=300, bbox_inches='tight')
plt.show()

# Combine user and post embeddings
all_embeddings = np.vstack([user_emb, post_emb])
node_types = ['user'] * len(user_emb) + ['post'] * len(post_emb)

# Reduce to 2D
pca_all = PCA(n_components=2, random_state=42)
all_emb_2d = pca_all.fit_transform(all_embeddings)

# Plot
plt.figure(figsize=(14, 10))
user_mask = np.array(node_types) == 'user'
post_mask = np.array(node_types) == 'post'

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