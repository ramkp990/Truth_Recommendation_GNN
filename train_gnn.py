# Trains a temporally filtered heterogeneous graph neural network with interaction-type weighting to recommend social media posts, evaluating performance using Recall@10 and NDCG@10 on future engagements while preventing temporal leakage.

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
import random

# ----------------------------
# Load processed data
# ----------------------------
# Load preprocessed graph data containing node features, indices, and activity records
data = torch.load("synthetic_processed_with_semantics.pt", weights_only=False)
activity_sub = data['activity_sub']        # Activity dataframe with posts and interactions
user_to_idx = data['user_to_idx']          # Mapping: user ID → node index (0 to num_users-1)
post_to_idx = data['post_to_idx']          # Mapping: post ID → node index (num_users to num_users+num_posts-1)
x = data['x']                              # Combined node feature matrix [users; posts]

num_users = data['num_users']              # Total number of user nodes
num_posts = data['num_posts']              # Total number of post nodes

# ----------------------------
# Temporal split (chronological 80/10/10 split for training/validation/test)
# ----------------------------
# Sort activities chronologically to simulate real-world temporal dynamics
activity_sorted = activity_sub.sort_values("timestamp").reset_index(drop=True)
n = len(activity_sorted)
train_end = int(0.8 * n)   # 80% for training
val_end = int(0.9 * n)     # 10% for validation
# Remaining 10% reserved for final testing (future engagements)
train_interactions = activity_sorted.iloc[:train_end]
val_interactions = activity_sorted.iloc[train_end:val_end]
test_interactions = activity_sorted.iloc[val_end:]

# ----------------------------
# Helper: build edge index safely with validation
# ----------------------------
def build_edge_index_safe(df, user_to_idx, post_to_idx):
    """
    Convert interaction dataframe to edge index tensors with safety checks.
    
    Args:
        df: DataFrame containing interactions with engager, target_user, post_id
        user_to_idx: Mapping from user ID strings to node indices
        post_to_idx: Mapping from post ID strings to global node indices
    
    Returns:
        tuple: (engagement_edge_index, authorship_edge_index)
            - engagement: [2, num_edges] tensor where row0=users, row1=posts (global indices)
            - authorship: [2, num_edges] tensor where row0=posts (global), row1=authors
    """
    engager, post_global, target_user = [], [], []
    for _, row in df.iterrows():
        # Map string IDs to node indices (skip unmapped entities)
        u_eng = user_to_idx.get(row["engager"])
        u_tgt = user_to_idx.get(row["target_user"])
        p_global = post_to_idx.get(row["post_id"])
        if u_eng is not None and u_tgt is not None and p_global is not None:
            engager.append(u_eng)
            post_global.append(p_global)
            target_user.append(u_tgt)
    
    # Convert lists to tensors
    engager = torch.tensor(engager, dtype=torch.long)
    post_global = torch.tensor(post_global, dtype=torch.long)
    target_user = torch.tensor(target_user, dtype=torch.long)
    
    # Construct edge index tensors: [2, num_edges]
    engage_edge = torch.stack([engager, post_global], dim=0)
    author_edge = torch.stack([post_global, target_user], dim=0)
    return engage_edge, author_edge

# Build test edges (training/validation edges built but unused here - likely for completeness)
_, _ = build_edge_index_safe(train_interactions, user_to_idx, post_to_idx)
_, _ = build_edge_index_safe(val_interactions, user_to_idx, post_to_idx)
test_pos_edges, _ = build_edge_index_safe(test_interactions, user_to_idx, post_to_idx)

# ----------------------------
# Build HeteroData graph (FIRST ATTEMPT - later overwritten)
# ----------------------------
# Note: This graph construction is immediately overwritten below - appears to be redundant code
graph = HeteroData()

# Node features (original 387-dim features before projection)
graph['user'].x = x[:num_users]      # [U, 387] - first U rows are user features
graph['post'].x = x[num_users:]      # [P, 387] - remaining rows are post features

# Edge 1: social relationships (user → user follow edges)
src, dst = data['edge_index_social']
graph['user', 'social', 'user'].edge_index = torch.stack([src, dst], dim=0)

# Edge 2: engagement edges (user → post) - using full dataset (not temporally filtered)
src, dst = data['edge_index_engage']
# Convert global post indices to local indices (0 to num_posts-1)
post_local = dst - num_users
# Safety mask to ensure indices are within valid ranges
mask = (src < num_users) & (post_local >= 0) & (post_local < num_posts)
graph['user', 'engages', 'post'].edge_index = torch.stack([src[mask], post_local[mask]], dim=0)

# Edge 3: authorship edges (post → user)
src, dst = data['edge_index_author']
post_local = src - num_users
mask = (post_local >= 0) & (post_local < num_posts) & (dst < num_users)
graph['post', 'authored_by', 'user'].edge_index = torch.stack([post_local[mask], dst[mask]], dim=0)

# Add reverse engagement edges for message passing (post → user)
graph['post', 'rev_engages', 'user'].edge_index = graph['user', 'engages', 'post'].edge_index.flip(0)

# ----------------------------
# REBUILD GRAPH WITH TEMPORAL FILTERING (CORRECT APPROACH)
# ----------------------------
# Overwrite previous graph with temporally filtered version (training edges only)
graph = HeteroData()

# Node features (same as before)
graph['user'].x = x[:num_users]
graph['post'].x = x[num_users:]

# Social edges (assumed static - include all follow relationships)
src, dst = data['edge_index_social']
graph['user', 'social', 'user'].edge_index = torch.stack([src, dst], dim=0)

# Engagement edges: ONLY from training period (temporal filtering to prevent leakage)
train_engage_edges, _ = build_edge_index_safe(train_interactions, user_to_idx, post_to_idx)
# Convert global post indices to local indices for graph construction
train_post_local = train_engage_edges[1] - num_users
mask = (train_engage_edges[0] < num_users) & (train_post_local >= 0) & (train_post_local < num_posts)
graph['user', 'engages', 'post'].edge_index = torch.stack([
    train_engage_edges[0][mask], 
    train_post_local[mask]
], dim=0)

# Authorship edges: ONLY for posts created in training period
# (Ensures no future information leaks into training graph)
train_posts = set(train_interactions["post_id"])
train_authorship = activity_sub[activity_sub["post_id"].isin(train_posts)]
# Note: Authorship edges not explicitly rebuilt here - relies on engagement edges above

# Add reverse engagement edges for message passing (post → user)
graph['post', 'rev_engages', 'user'].edge_index = graph['user', 'engages', 'post'].edge_index.flip(0)

# ----------------------------
# Updated WeightedRGCN Model (simplified version with 2 message types)
# ----------------------------
class WeightedRGCN(torch.nn.Module):
    """
    Weighted Relational Graph Convolutional Network with two message types:
    1. Direct engagement signal (post → user via reverse engagement edges)
    2. Social graph signal (user → user via follow relationships)
    
    Note: Author-follow signal removed compared to previous version (commented out)
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Message passing layers for different edge types
        self.msg_direct = SAGEConv((-1, -1), hidden_dim)   # user ← post (engagement)
        self.msg_social = SAGEConv((-1, -1), hidden_dim)   # user ← user (social)
        self.post_update = SAGEConv((-1, -1), hidden_dim)  # post ← user (engagement)
        
        # Fixed weights for combining message types (empirically tuned)
        self.w_direct = 1.0    # Strongest weight for direct engagement signal
        self.w_social = 0.75   # Secondary weight for social graph signal

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through heterogeneous graph:
        1. Aggregate direct engagement messages (posts → users)
        2. Aggregate social graph messages (users → users)
        3. Combine signals with weighted sum and ReLU activation
        4. Update post representations using current user embeddings
        """
        user_x, post_x = x_dict['user'], x_dict['post']
        
        # User update: combine direct engagement and social signals
        msg_direct = self.msg_direct(
            (post_x, user_x),
            edge_index_dict[('post', 'rev_engages', 'user')]
        )
        msg_social = self.msg_social(
            (user_x, user_x),
            edge_index_dict[('user', 'social', 'user')]
        )
        
        # Weighted combination of messages with ReLU activation
        user_out = F.relu(
            self.w_direct * msg_direct +
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

# ----------------------------
# Training setup
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WeightedRGCN(hidden_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy for link prediction

# Move graph to computation device and prepare input dictionaries
graph = graph.to(device)
x_dict = {'user': graph['user'].x, 'post': graph['post'].x}
train_edge_index = graph['user', 'engages', 'post'].edge_index  # Training edges only

num_users = graph['user'].num_nodes
num_posts = graph['post'].num_nodes

print(f"Training on {train_edge_index.shape[1]} positive edges")
print(f"Users: {num_users}, Posts: {num_posts}")

# ----------------------------
# Interaction-type weighting setup
# ----------------------------
# Precompute weights based on interaction type: Quotes weighted 3x more than replies/posts
# (Reflects higher semantic value of quote interactions vs simple replies)
global_post_to_interaction = {}
for _, row in train_interactions.iterrows():
    local_post_id = row["post_id"]
    global_post_id = post_to_idx[local_post_id]  # Convert local ID to global node index
    global_post_to_interaction[global_post_id] = row["interaction"]

# Create lookup tensor for fast weight retrieval during training
max_global_post_id = max(post_to_idx.values())
interaction_type_tensor = torch.zeros(max_global_post_id + 1, dtype=torch.float32, device=device)
for gid, inter in global_post_to_interaction.items():
    weight = 3.0 if inter == "QT" else 1.0  # QT=quote gets 3x weight, RE/POST get 1x
    interaction_type_tensor[gid] = weight

# ----------------------------
# Training Function with interaction-type weighting
# ----------------------------
def train():
    """
    Single training step with:
    1. Forward pass through GNN to get embeddings
    2. Positive edge scoring (with interaction-type weighting)
    3. Negative sampling (1:1 ratio)
    4. Weighted loss computation and backpropagation
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass through GNN
    out = model(x_dict, graph.edge_index_dict)
    user_emb = out['user']  # [num_users, 64]
    post_emb = out['post']  # [num_posts, 64]

    # Positive edges from training set
    pos_u, pos_p = train_edge_index  # pos_p is LOCAL post index (0 to num_posts-1)

    # Compute positive scores via dot product
    pos_scores = (user_emb[pos_u] * post_emb[pos_p]).sum(dim=1)  # [num_pos]

    # --- Apply weights based on interaction type ---
    # Convert local post index → global post ID (for weight lookup)
    pos_global_post_ids = pos_p + num_users  # posts start at index = num_users
    
    # Fetch weights using precomputed tensor (shape: [num_pos])
    pos_weights = interaction_type_tensor[pos_global_post_ids]

    # Negative sampling: random posts for each positive edge
    neg_p = torch.randint(0, num_posts, (pos_p.size(0),), device=device)
    neg_scores = (user_emb[pos_u] * post_emb[neg_p]).sum(dim=1)

    # Compute weighted loss
    pos_loss = criterion(pos_scores, torch.ones_like(pos_scores))
    neg_loss = criterion(neg_scores, torch.zeros_like(neg_scores))

    # Weight positive losses by interaction type (quotes penalized more for errors)
    weighted_pos_loss = (pos_weights * pos_loss).mean()
    loss = weighted_pos_loss + neg_loss  # Combined loss

    loss.backward()
    optimizer.step()
    return loss.item()

# ----------------------------
# Evaluation Function (Recall@10, NDCG@10)
# ----------------------------
def evaluate(test_edges, user_emb, post_emb, K=10):
    """
    Evaluate recommendation quality using:
    - Recall@K: proportion of relevant items in top-K recommendations
    - NDCG@K: normalized discounted cumulative gain (position-aware ranking metric)
    
    Args:
        test_edges: [2, num_edges] tensor of test engagements (users → posts)
        user_emb: [num_users, dim] user embeddings
        post_emb: [num_posts, dim] post embeddings
        K: cutoff for top-K evaluation
    
    Returns:
        tuple: (mean_recall, mean_ndcg) across all test users
    """
    from collections import defaultdict
    import numpy as np
    from sklearn.metrics import ndcg_score

    # Group test engagements by user
    user_test_posts = defaultdict(list)
    candidate_posts = set()  # All posts appearing in test set
    
    for i in range(test_edges.shape[1]):
        u = test_edges[0, i].item()
        p_global = test_edges[1, i].item()
        p_local = p_global - num_users  # Convert to local post index
        user_test_posts[u].append(p_local)
        candidate_posts.add(p_local)

    # Convert candidate set to sorted tensor for scoring
    candidate_posts = sorted(candidate_posts)
    candidate_posts = torch.tensor(candidate_posts, device=device)
    
    recall_list, ndcg_list = [], []
    
    # Evaluate each user with test engagements
    for user_id in user_test_posts:
        if user_id >= num_users:  # Skip invalid user indices
            continue
        true_posts = user_test_posts[user_id]
        if not true_posts:  # Skip users with no test engagements
            continue

        # Score all candidate posts for this user
        scores = torch.mm(
            user_emb[user_id].unsqueeze(0),  # [1, dim]
            post_emb[candidate_posts].T      # [dim, num_candidates]
        ).squeeze(0)  # [num_candidates]

        # Get top-K recommendations
        topk_idx = torch.topk(scores, min(K, len(scores)))[1]
        topk_posts = candidate_posts[topk_idx].cpu().tolist()

        # Compute Recall@K
        hits = len(set(topk_posts) & set(true_posts))
        recall = hits / len(true_posts)
        recall_list.append(recall)

        # Compute NDCG@K (position-aware metric)
        relevance = torch.zeros(len(candidate_posts), device=device)
        for p in true_posts:
            if p in candidate_posts:
                idx = (candidate_posts == p).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    relevance[idx] = 1.0  # Binary relevance
        
        # Only compute NDCG if there are relevant items
        if relevance.sum() > 0:
            ndcg = ndcg_score(
                relevance.cpu().numpy().reshape(1, -1),  # Ground truth relevance
                scores.cpu().numpy().reshape(1, -1),     # Predicted scores
                k=K
            )
            ndcg_list.append(ndcg)

    # Return mean metrics across users
    return np.mean(recall_list), np.mean(ndcg_list)

# ----------------------------
# Training Loop with Loss Tracking and Early Stopping
# ----------------------------
print("\n=== START TRAINING ===")
best_recall = 0.0
patience = 20  # Early stopping patience (epochs without improvement)
patience_counter = 0

# Track metrics for analysis/plotting
train_losses = []
val_recalls = []
val_ndcgs = []
epochs_list = []

for epoch in range(1, 201):  # Maximum 200 epochs
    loss = train()
    train_losses.append(loss)
    
    # Evaluate every 10 epochs
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            out = model(x_dict, graph.edge_index_dict)
            user_emb = out['user']
            post_emb = out['post']
            
            # Move test edges to device for evaluation
            test_edges_device = test_pos_edges.to(device)
            recall, ndcg = evaluate(test_edges_device, user_emb, post_emb, K=10)
            
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Recall@10: {recall:.4f} | NDCG@10: {ndcg:.4f}")
            
            # Store metrics for tracking
            epochs_list.append(epoch)
            val_recalls.append(recall)
            val_ndcgs.append(ndcg)
            
            # Early stopping with model checkpointing
            if recall > best_recall:
                best_recall = recall
                patience_counter = 0
                torch.save(model.state_dict(), "best_rgcn_model.pt")  # Save best model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

# ----------------------------
# Final Evaluation with Best Model
# ----------------------------
print("\n=== FINAL EVALUATION ===")
model.load_state_dict(torch.load("best_rgcn_model.pt"))
model.eval()
with torch.no_grad():
    out = model(x_dict, graph.edge_index_dict)
    user_emb = out['user']
    post_emb = out['post']
    test_edges_device = test_pos_edges.to(device)
    recall, ndcg = evaluate(test_edges_device, user_emb, post_emb, K=10)
    print(f"Best Recall@10: {recall:.4f}")
    print(f"Best NDCG@10: {ndcg:.4f}")

# Save final embeddings for downstream tasks
torch.save({
    'user_emb': user_emb.cpu(),
    'post_emb': post_emb.cpu(),
    'user_to_idx': user_to_idx,
    'num_users': num_users,
}, "higgs_embeddings_trained.pt")

# ----------------------------
# Save Training Metrics to JSON
# ----------------------------
import json
metrics = {
    'epochs': epochs_list,
    'train_losses': train_losses[9::10],  # Sample every 10th loss to match evaluation frequency
    'val_recalls': val_recalls,
    'val_ndcgs': val_ndcgs,
    'final_recall': float(recall),
    'final_ndcg': float(ndcg)
}

with open('training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Training metrics saved to 'training_metrics.json'")
print(f"   Final Recall@10: {recall:.4f}")
print(f"   Final NDCG@10: {ndcg:.4f}")

# ----------------------------
# Save Detailed Recommendation Results to Text File
# ----------------------------
results_file = "recommendation_results.txt"
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("=== RECOMMENDATION RESULTS ===\n")
    f.write(f"Final Recall@10: {recall:.4f}\n")
    f.write(f"Final NDCG@10: {ndcg:.4f}\n\n")
    
    # Sample recommendations for qualitative analysis
    f.write("🔍 SAMPLE RECOMMENDATIONS (Top 3 per user):\n")
    test_users = list(set(test_edges_device[0].cpu().numpy()))
    # Randomly sample up to 5 test users for inspection
    test_users_sample = random.sample(test_users, min(5, len(test_users)))

    # Build tweet lookup dictionary for displaying recommendations
    activity_sub = data['activity_sub']
    tweet_lookup = {}
    for _, row in activity_sub.iterrows():
        global_post_id = post_to_idx[row["post_id"]]
        tweet_lookup[global_post_id] = row["tweet"]

    # Generate and save recommendations for sampled users
    for user_global in test_users_sample:
        user_local = user_global
        # Get true future engagements for this user
        true_posts_global = test_edges_device[1][test_edges_device[0] == user_global].cpu().numpy()
        
        # Build candidate pool from all test-period posts
        candidate_posts_global = set(test_edges_device[1].cpu().numpy())
        candidate_posts_local = [int(gid - num_users) for gid in candidate_posts_global]
        candidate_posts_global = list(candidate_posts_global)
        
        # Score candidates and get top-3
        scores = torch.mm(
            user_emb[user_local].unsqueeze(0),
            post_emb[torch.tensor(candidate_posts_local, device=device)].T
        ).squeeze(0)
        
        topk_idx = torch.topk(scores, min(3, len(scores)))[1]
        top_posts_local = [candidate_posts_local[i] for i in topk_idx]
        top_posts_global = [pid + num_users for pid in top_posts_local]
        top_scores = scores[topk_idx].cpu().numpy()
        
        # Write results to file
        f.write(f"\nUser {user_global}:\n")
        f.write("  ✅ True future engagements:\n")
        for p in true_posts_global:
            tweet = tweet_lookup.get(p, "[MISSING]")
            f.write(f"    - {tweet}\n")
        
        f.write("  🎯 Top recommendations (with scores):\n")
        for i, (p, score) in enumerate(zip(top_posts_global, top_scores)):
            tweet = tweet_lookup.get(p, "[MISSING]")
            f.write(f"    [{i+1}] Score: {score:.4f} | {tweet}\n")

print(f"\n✅ Full results saved to '{results_file}'")
print("   (Contains complete tweets and recommendation scores)")
