# Evaluates a trained heterogeneous graph neural network on the top 10 most active test users using Recall@10 and NDCG@10 metrics to measure recommendation quality for future engagements.

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import ndcg_score

# ----------------------------
# 1. Load processed data and test interactions
# ----------------------------
print("Loading data...")
# Load preprocessed graph data containing node features, indices, and activity records
data = torch.load("synthetic_processed_with_semantics.pt", weights_only=False)
activity_sub = data['activity_sub']        # Activity dataframe with posts and interactions
user_to_idx = data['user_to_idx']          # Mapping: user ID → node index (0 to num_users-1)
post_to_idx = data['post_to_idx']          # Mapping: post ID → node index (num_users to num_users+num_posts-1)
x = data['x']                              # Combined node feature matrix [users; posts]

num_users = data['num_users']              # Total number of user nodes
num_posts = data['num_posts']              # Total number of post nodes

# Temporal split: 80% train, 10% validation, 10% test (chronological split)
activity_sorted = activity_sub.sort_values("timestamp").reset_index(drop=True)
n = len(activity_sorted)
train_end = int(0.8 * n)
val_end = int(0.9 * n)
test_interactions = activity_sorted.iloc[val_end:]  # Last 10% for testing

# Build test edge index tensor from test interactions
def build_test_edges(df, user_to_idx, post_to_idx):
    """
    Convert test interactions dataframe to edge index tensor format.
    
    Args:
        df: DataFrame with test interactions
        user_to_idx: User ID to index mapping
        post_to_idx: Post ID to index mapping
    
    Returns:
        torch.Tensor: Edge index of shape [2, num_edges] where row 0 = users, row 1 = posts
    """
    engager, post_global = [], []
    for _, row in df.iterrows():
        u_eng = user_to_idx.get(row["engager"])      # Map engager to node index
        p_global = post_to_idx.get(row["post_id"])   # Map post to global node index
        if u_eng is not None and p_global is not None:  # Skip unmapped entities
            engager.append(u_eng)
            post_global.append(p_global)
    return torch.tensor([engager, post_global], dtype=torch.long)

test_pos_edges = build_test_edges(test_interactions, user_to_idx, post_to_idx)

# Identify top 10 most active users in test set (by engagement count)
user_test_counts = defaultdict(int)
for i in range(test_pos_edges.shape[1]):
    user_id = test_pos_edges[0, i].item()  # Extract user index from edge
    user_test_counts[user_id] += 1         # Count engagements per user

# Sort users by test engagement count and select top 10
top_10_users = sorted(user_test_counts.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"Top 10 most active test users:")
for i, (user_id, count) in enumerate(top_10_users):
    print(f"  {i+1}. User {user_id}: {count} engagements")

# ----------------------------
# 2. Rebuild graph for inference
# ----------------------------
# Construct heterogeneous graph object for GNN inference
graph = HeteroData()
graph['user'].x = x[:num_users]    # User node features (first num_users rows)
graph['post'].x = x[num_users:]    # Post node features (remaining rows)

# Add social edges: user → user (follow relationships)
src, dst = data['edge_index_social']
graph['user', 'social', 'user'].edge_index = torch.stack([src, dst], dim=0)

# Add engagement edges: user → post (who interacted with which post)
# Convert global post indices to local post indices (0 to num_posts-1)
src, dst = data['edge_index_engage']
post_local = dst - num_users
mask = (src < num_users) & (post_local >= 0) & (post_local < num_posts)  # Validity check
graph['user', 'engages', 'post'].edge_index = torch.stack([src[mask], post_local[mask]], dim=0)

# Add authorship edges: post → user (which user authored which post)
src, dst = data['edge_index_author']
post_local = src - num_users
mask = (post_local >= 0) & (post_local < num_posts) & (dst < num_users)
graph['post', 'authored_by', 'user'].edge_index = torch.stack([post_local[mask], dst[mask]], dim=0)

# Add author-follow edges: post → user (users who follow the post's author)
# Note: This edge type may not exist in all saved datasets (handled with fallback)
try:
    src, dst = data['edge_index_followed_by']
    if src.numel() > 0:  # Check if edge tensor is non-empty
        post_local = src - num_users
        mask = (post_local >= 0) & (post_local < num_posts) & (dst < num_users)
        graph['post', 'followed_by', 'user'].edge_index = torch.stack([post_local[mask], dst[mask]], dim=0)
    else:
        graph['post', 'followed_by', 'user'].edge_index = torch.empty((2, 0), dtype=torch.long)
except KeyError:
    # Fallback if edge_index_followed_by not in saved data
    graph['post', 'followed_by', 'user'].edge_index = torch.empty((2, 0), dtype=torch.long)

# Add reverse engagement edges: post → user (for message passing from posts to users)
graph['post', 'rev_engages', 'user'].edge_index = graph['user', 'engages', 'post'].edge_index.flip(0)

# ----------------------------
# 3. Load trained model
# ----------------------------
# Define WeightedRGCN architecture matching training configuration
# Note: This version includes author-follow signal (followed_by edges) not present in earlier versions
class WeightedRGCN(torch.nn.Module):
    """
    Weighted Relational Graph Convolutional Network with three message types:
    1. Direct engagement (post → user via rev_engages)
    2. Author-follow signal (post → user via followed_by)
    3. Social graph signal (user → user via social edges)
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Message passing layers for different edge types
        self.msg_direct = SAGEConv((-1, -1), hidden_dim)   # user ← post (engagement)
        self.msg_author = SAGEConv((-1, -1), hidden_dim)   # user ← post (follows author)
        self.msg_social = SAGEConv((-1, -1), hidden_dim)   # user ← user (social)
        self.post_update = SAGEConv((-1, -1), hidden_dim)  # post ← user (engagement)
        
        # Fixed weights for combining message types (empirically tuned)
        self.w_direct = 1.75   # Strongest weight for direct engagement signal
        self.w_author = 0.7    # Medium weight for author-follow signal
        self.w_social = 0.3    # Weakest weight for social graph signal

    def forward(self, x_dict, edge_index_dict):
        user_x, post_x = x_dict['user'], x_dict['post']
        
        # Aggregate three message types for user representation update
        msg_direct = self.msg_direct(
            (post_x, user_x),
            edge_index_dict[('post', 'rev_engages', 'user')]
        )
        msg_author = self.msg_author(
            (post_x, user_x),
            edge_index_dict[('post', 'followed_by', 'user')]
        )
        msg_social = self.msg_social(
            (user_x, user_x),
            edge_index_dict[('user', 'social', 'user')]
        )
        
        # Weighted combination of messages with ReLU activation
        user_out = F.relu(
            self.w_direct * msg_direct +
            self.w_author * msg_author +
            self.w_social * msg_social
        )
        
        # Update post representations using current user embeddings
        post_out = F.relu(
            self.post_update(
                (user_x, post_x),
                edge_index_dict[('user', 'engages', 'post')]
            )
        )
        
        return {'user': user_out, 'post': post_out}

# Initialize model and load trained weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WeightedRGCN(hidden_dim=64).to(device)
model.load_state_dict(torch.load("best_rgcn_model.pt", weights_only=True))
model.eval()  # Set to evaluation mode (disable dropout/batchnorm)

# Move graph to computation device
graph = graph.to(device)
x_dict = {'user': graph['user'].x, 'post': graph['post'].x}

# ----------------------------
# 4. Compute embeddings
# ----------------------------
# Generate node embeddings using trained GNN (no gradient computation)
with torch.no_grad():
    out = model(x_dict, graph.edge_index_dict)
    user_emb = out['user']  # [num_users, hidden_dim] user embeddings
    post_emb = out['post']  # [num_posts, hidden_dim] post embeddings

# ----------------------------
# 5. Evaluate TOP 10 MOST ACTIVE USERS
# ----------------------------
test_edges_device = test_pos_edges.to(device)

# Build candidate pool: all posts appearing in test interactions
candidate_posts_global = set(test_edges_device[1].cpu().numpy())  # Global post indices
candidate_posts_local = sorted([int(p - num_users) for p in candidate_posts_global])  # Convert to local indices
candidate_posts_local = torch.tensor(candidate_posts_local, device=device)

# Build tweet lookup dictionary for displaying recommendations
tweet_lookup = {}
for _, row in activity_sub.iterrows():
    global_id = post_to_idx[row["post_id"]]  # Map local post ID to global node index
    tweet_lookup[global_id] = row["tweet"]

# Store evaluation results per user
results = []

print("\n" + "="*60)
print("EVALUATION FOR TOP 10 MOST ACTIVE TEST USERS")
print("="*60)

# Evaluate each of the top 10 most active test users
for rank, (user_id, true_count) in enumerate(top_10_users, 1):
    # Extract true positive posts for this user from test set
    true_mask = (test_edges_device[0] == user_id)
    true_posts_global = test_edges_device[1][true_mask].cpu().numpy()
    true_posts_local = [int(p - num_users) for p in true_posts_global]  # Convert to local indices
    
    # Skip if no true engagements (shouldn't happen for top users)
    if not true_posts_local:
        continue
        
    # Score all candidate posts using dot product (cosine similarity since embeddings normalized)
    scores = torch.mm(
        user_emb[user_id].unsqueeze(0),      # [1, hidden_dim]
        post_emb[candidate_posts_local].T    # [hidden_dim, num_candidates]
    ).squeeze(0)  # [num_candidates]
    
    # Get top-10 recommendations
    K = 10
    topk_idx = torch.topk(scores, min(K, len(scores)))[1]  # Indices of top-K scores
    topk_posts_local = candidate_posts_local[topk_idx].cpu().tolist()  # Convert to local post IDs
    
    # Compute Recall@10: proportion of true engagements in top-10 recommendations
    hits = len(set(topk_posts_local) & set(true_posts_local))
    recall_at_10 = hits / len(true_posts_local)
    
    # Compute NDCG@10: graded ranking metric considering position of relevant items
    relevance = torch.zeros(len(candidate_posts_local), device=device)
    for p in true_posts_local:
        if p in candidate_posts_local:
            idx = (candidate_posts_local == p).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                relevance[idx] = 1.0  # Binary relevance (engaged = 1, not engaged = 0)
    
    ndcg_at_10 = 0.0
    if relevance.sum() > 0:  # Only compute if there are relevant items
        ndcg_at_10 = ndcg_score(
            relevance.cpu().numpy().reshape(1, -1),   # Ground truth relevance
            scores.cpu().numpy().reshape(1, -1),      # Predicted scores
            k=K
        )
    
    # Store results
    results.append((user_id, true_count, recall_at_10, ndcg_at_10))
    
    # Print evaluation metrics
    print(f"\n{rank}. USER {user_id} (Test Engagements: {true_count})")
    print(f"   Recall@10: {recall_at_10:.4f} | NDCG@10: {ndcg_at_10:.4f}")
    
    # Show qualitative examples for first 3 users only (avoid clutter)
    if rank <= 3:
        print(f"   ✅ True engagements (first 3):")
        for p_global in true_posts_global[:3]:
            tweet = tweet_lookup.get(p_global, "[MISSING]")
            print(f"     - {tweet[:60]}...")
        
        print(f"   🎯 Top recommendations (first 3):")
        top_posts_global = [int(p + num_users) for p in topk_posts_local[:3]]  # Convert back to global indices
        for p_global in top_posts_global:
            tweet = tweet_lookup.get(p_global, "[MISSING]")
            print(f"     - {tweet[:60]}...")

# ----------------------------
# 6. Summary Statistics
# ----------------------------
# Aggregate metrics across all 10 evaluated users
recall_scores = [r[2] for r in results]
ndcg_scores = [r[3] for r in results]

print("\n" + "="*60)
print("SUMMARY FOR TOP 10 MOST ACTIVE USERS")
print("="*60)
print(f"Average Recall@10: {np.mean(recall_scores):.4f}")
print(f"Average NDCG@10:  {np.mean(ndcg_scores):.4f}")
print(f"Users with Recall@10 > 0: {sum(1 for r in recall_scores if r > 0)} / 10")

