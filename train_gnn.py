'''
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv  # SAGEConv works well for heterogeneous graphs
import random

# ----------------------------
# Load and prepare data (your code)
# ----------------------------
data = torch.load("synthetic_processed_with_semantics.pt")
data = torch.load("synthetic_processed_with_semantics.pt", weights_only=False)
activity_sub = data['activity_sub']

# DIAGNOSTIC: What are the actual post_id values?
print("First 5 post_id in activity_sub:", activity_sub["post_id"].head().values)
print("post_id min:", activity_sub["post_id"].min())
print("post_id max:", activity_sub["post_id"].max())
print("Expected post_id range: 0 to", len(activity_sub) - 1)

# Check post_to_idx keys
post_to_idx = data['post_to_idx']
print("First 5 keys in post_to_idx:", list(post_to_idx.keys())[:5])
print("Are post_id values in post_to_idx keys?", 
      activity_sub["post_id"].isin(post_to_idx.keys()).all())
user_to_idx = data['user_to_idx']
post_to_idx = data['post_to_idx']
x = data['x']



num_users = len(user_to_idx)
post_features = x[num_users:]  # [P, 387]

# Boost semantic part (dims 3-386)
semantic_boost = 2.0
post_features[:, 3:] *= semantic_boost

# Rebuild x
x = torch.cat([x[:num_users], post_features], dim=0)



edge_index_social = data['edge_index_social']
activity_sub = data['activity_sub']

#activity_sorted = activity_sub
activity_sorted = activity_sub.sort_values("timestamp").reset_index(drop=True)
#activity_sorted["post_id"] = activity_sorted.index

n = len(activity_sorted)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_interactions = activity_sorted.iloc[:train_end]
val_interactions = activity_sorted.iloc[train_end:val_end]
test_interactions = activity_sorted.iloc[val_end:]


def build_edge_index_safe(df, user_to_idx, post_to_idx, num_users):
    engager = []
    post_global = []
    target_user = []

    for _, row in df.iterrows():
        u_eng = user_to_idx.get(row["engager"])
        u_tgt = user_to_idx.get(row["target_user"])
        p_local = row["post_id"]  # this is 0-9999
        p_global = post_to_idx.get(p_local)
        #print(p_global)
        if u_eng is not None and u_tgt is not None and p_global is not None:
            engager.append(u_eng)
            post_global.append(p_global)
            target_user.append(u_tgt)

    engager = torch.tensor(engager, dtype=torch.long)
    post_global = torch.tensor(post_global, dtype=torch.long)
    target_user = torch.tensor(target_user, dtype=torch.long)

    engage_edge = torch.stack([engager, post_global], dim=0)
    author_edge = torch.stack([post_global, target_user], dim=0)
    return engage_edge, author_edge

# Use it
U = len(user_to_idx)
edge_index_engage_train, edge_index_author_train = build_edge_index_safe(
    train_interactions, user_to_idx, post_to_idx, U
)
val_pos_edges, _ = build_edge_index_safe(val_interactions, user_to_idx, post_to_idx, U)
test_pos_edges, _ = build_edge_index_safe(test_interactions, user_to_idx, post_to_idx, U)


print("\n=== DEBUG TEST EDGES ===")
print("test_pos_edges shape:", test_pos_edges.shape)
print("First 5 user (src):", test_pos_edges[0, :5].numpy())
print("First 5 post (dst):", test_pos_edges[1, :5].numpy())

U = len(user_to_idx)
print(U)
print(f"User ID range: 0 to {U - 1}")
print(f"Expected post ID range: {U} to {U + 9999}")

# Check for out-of-range posts
post_ids = test_pos_edges[1].numpy()
invalid_mask = (post_ids < U) | (post_ids >= U + 10000)
if invalid_mask.any():
    print("⚠️  Invalid post IDs found:")
    print("Sample invalid post IDs:", post_ids[invalid_mask][:5])
else:
    print("✅ All post IDs are in valid range.")
# ----------------------------
# Build HeteroData graph
# ----------------------------
# ----------------------------
# Build HeteroData with LOCAL indices
# ----------------------------
graph = HeteroData()


# 1. Node features (already split logically)
num_users = len(user_to_idx)
num_posts = len(post_to_idx)

graph['user'].x = x[:num_users]      # [4967, 3]
graph['post'].x = x[num_users:]      # [10000, 3]

# 2. Convert edge indices to LOCAL IDs
# Social: user (global) → user (global) → convert to local (same as global for users)
# But ensure all user IDs are in [0, num_users)
src, dst = edge_index_social
# Filter edges to only include valid users (should be all, but safe)
mask = (src < num_users) & (dst < num_users)
graph['user', 'social', 'user'].edge_index = torch.stack([src[mask], dst[mask]], dim=0)

# Engagement: user (global) → post (global)
# Convert post global ID → local post ID: global_id - num_users
src, dst = edge_index_engage_train
post_local = dst - num_users
mask = (src < num_users) & (post_local >= 0) & (post_local < num_posts)
graph['user', 'engages', 'post'].edge_index = torch.stack([src[mask], post_local[mask]], dim=0)

# Authorship: post (global) → user (global)
src, dst = edge_index_author_train
post_local = src - num_users
mask = (post_local >= 0) & (post_local < num_posts) & (dst < num_users)
graph['post', 'authored_by', 'user'].edge_index = torch.stack([post_local[mask], dst[mask]], dim=0)

# Optional: add reverse engagement edge
# PyG can auto-add reverse, or do manually:
rev_engage = graph['user', 'engages', 'post'].edge_index.flip(0)  # [post, user]
graph['post', 'rev_engages', 'user'].edge_index = rev_engage


# ----------------------------
# Define 1-layer R-GCN (via HeteroConv)
# ----------------------------
class SimpleRGCN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv1 = HeteroConv({
            ('user', 'social', 'user'): SAGEConv((-1, -1), hidden_dim),
            ('user', 'engages', 'post'): SAGEConv((-1, -1), hidden_dim),
            ('post', 'rev_engages', 'user'): SAGEConv((-1, -1), hidden_dim),
            ('post', 'authored_by', 'user'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('user', 'social', 'user'): SAGEConv((-1, -1), hidden_dim),
            ('user', 'engages', 'post'): SAGEConv((-1, -1), hidden_dim),
            ('post', 'rev_engages', 'user'): SAGEConv((-1, -1), hidden_dim),
            ('post', 'authored_by', 'user'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')
        self.hidden_dim = hidden_dim

    def forward(self, x_dict, edge_index_dict):
        # First layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Second layer (optional but helpful)
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        return x_dict

class WeightedRGCN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # For updating USERS
        self.user_from_social = SAGEConv((-1, -1), hidden_dim)   # user ← user (social)
        self.user_from_posts = SAGEConv((-1, -1), hidden_dim)    # user ← post (rev_engages)
        
        # For updating POSTS (optional; you can even skip this!)
        self.post_from_users = SAGEConv((-1, -1), hidden_dim)    # post ← user (engages)
        
        # Weights: direct engagement > social influence
        self.w_direct = torch.nn.Parameter(torch.tensor(1.0))
        self.w_social = torch.nn.Parameter(torch.tensor(0.3))

    def forward(self, x_dict, edge_index_dict):
        user_x, post_x = x_dict['user'], x_dict['post']
        
        # Update USERS
        msg_social = self.user_from_social(
            (user_x, user_x),
            edge_index_dict[('user', 'social', 'user')]
        )
        msg_direct = self.user_from_posts(
            (post_x, user_x),
            edge_index_dict[('post', 'rev_engages', 'user')]  # ← you have this!
        )
        user_out = F.relu(self.w_social * msg_social + self.w_direct * msg_direct)
        
        # Update POSTS (optional)
        msg_engage = self.post_from_users(
            (user_x, post_x),
            edge_index_dict[('user', 'engages', 'post')]  # ← you have this!
        )
        post_out = F.relu(msg_engage)
        
        return {'user': user_out, 'post': post_out}
    
class WeightedRGCNFixed(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Message functions
        self.msg_direct = SAGEConv((-1, -1), hidden_dim)      # user ← post (engagement)
        self.msg_social = SAGEConv((-1, -1), hidden_dim)      # user ← user (follows)
        self.msg_author = SAGEConv((-1, -1), hidden_dim)      # user ← post (follows author)
        
        # FIXED weights (prioritize signals)
        self.w_direct = 1.0   # strongest: "I engaged with this"
        self.w_author = 0.7   # medium: "I follow the author"
        self.w_social = 0.3   # weakest: "My friend follows someone"

    def forward(self, x_dict, edge_index_dict):
        user_x, post_x = x_dict['user'], x_dict['post']
        
        # Direct engagement (user ← post via rev_engages)
        msg1 = self.msg_direct(
            (post_x, user_x),
            edge_index_dict[('post', 'rev_engages', 'user')]
        )
        
        # Follows author (user ← post via new edge)
        msg2 = self.msg_author(
            (post_x, user_x),
            edge_index_dict[('post', 'followed_by', 'user')]  # ← we'll build this
        )
        
        # Social influence (user ← user)
        msg3 = self.msg_social(
            (user_x, user_x),
            edge_index_dict[('user', 'social', 'user')]
        )
        
        # Combine with weights
        user_out = F.relu(
            self.w_direct * msg1 +
            self.w_author * msg2 +
            self.w_social * msg3
        )
        
        # Update posts (optional)
        post_out = F.relu(
            self.msg_direct(
                (user_x, post_x),
                edge_index_dict[('user', 'engages', 'post')]
            )
        )
        
        return {'user': user_out, 'post': post_out}

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import negative_sampling
import random

# ----------------------------
# Training Setup
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WeightedRGCNFixed(hidden_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

# Move graph to device
graph = graph.to(device)
x_dict = {'user': graph['user'].x, 'post': graph['post'].x}

# Get training engagement edges (user → post, local indices)
train_edge_index = graph['user', 'engages', 'post'].edge_index  # [2, num_train]

# Total number of users and posts
num_users = graph['user'].num_nodes
num_posts = graph['post'].num_nodes
'''



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
data = torch.load("synthetic_processed_with_semantics.pt", weights_only=False)
activity_sub = data['activity_sub']
user_to_idx = data['user_to_idx']
post_to_idx = data['post_to_idx']
x = data['x']

num_users = data['num_users']
num_posts = data['num_posts']

# ----------------------------
# Temporal split
# ----------------------------
activity_sorted = activity_sub.sort_values("timestamp").reset_index(drop=True)
n = len(activity_sorted)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_interactions = activity_sorted.iloc[:train_end]
val_interactions = activity_sorted.iloc[train_end:val_end]
test_interactions = activity_sorted.iloc[val_end:]

# ----------------------------
# Helper: build edge index safely
# ----------------------------
def build_edge_index_safe(df, user_to_idx, post_to_idx):
    engager, post_global, target_user = [], [], []
    for _, row in df.iterrows():
        u_eng = user_to_idx.get(row["engager"])
        u_tgt = user_to_idx.get(row["target_user"])
        p_global = post_to_idx.get(row["post_id"])
        if u_eng is not None and u_tgt is not None and p_global is not None:
            engager.append(u_eng)
            post_global.append(p_global)
            target_user.append(u_tgt)
    engager = torch.tensor(engager, dtype=torch.long)
    post_global = torch.tensor(post_global, dtype=torch.long)
    target_user = torch.tensor(target_user, dtype=torch.long)
    engage_edge = torch.stack([engager, post_global], dim=0)
    author_edge = torch.stack([post_global, target_user], dim=0)
    return engage_edge, author_edge

# Build test edges
_, _ = build_edge_index_safe(train_interactions, user_to_idx, post_to_idx)
_, _ = build_edge_index_safe(val_interactions, user_to_idx, post_to_idx)
test_pos_edges, _ = build_edge_index_safe(test_interactions, user_to_idx, post_to_idx)

# ----------------------------
# Build HeteroData graph
# ----------------------------
graph = HeteroData()

# Node features
graph['user'].x = x[:num_users]      # [U, 387]
graph['post'].x = x[num_users:]      # [P, 387]

# Edge 1: social (user → user)
src, dst = data['edge_index_social']
graph['user', 'social', 'user'].edge_index = torch.stack([src, dst], dim=0)

# Edge 2: engages (user → post) — for training
src, dst = data['edge_index_engage']
# Convert post global ID → local: global_id - num_users
post_local = dst - num_users
mask = (src < num_users) & (post_local >= 0) & (post_local < num_posts)
graph['user', 'engages', 'post'].edge_index = torch.stack([src[mask], post_local[mask]], dim=0)

# Edge 3: authored_by (post → user)
src, dst = data['edge_index_author']
post_local = src - num_users
mask = (post_local >= 0) & (post_local < num_posts) & (dst < num_users)
graph['post', 'authored_by', 'user'].edge_index = torch.stack([post_local[mask], dst[mask]], dim=0)


# Add reverse edges (for message passing)
graph['post', 'rev_engages', 'user'].edge_index = graph['user', 'engages', 'post'].edge_index.flip(0)

# ----------------------------
# Updated WeightedRGCN Model
# ----------------------------
class WeightedRGCN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.msg_direct = SAGEConv((-1, -1), hidden_dim)   # user ← post (engagement)
        #self.msg_author = SAGEConv((-1, -1), hidden_dim)   # user ← post (follows author)
        self.msg_social = SAGEConv((-1, -1), hidden_dim)   # user ← user (social)
        self.post_update = SAGEConv((-1, -1), hidden_dim)  # post ← user (engagement)
        
        # FIXED weights
        self.w_direct = 1.0   # strongest
        #self.w_author = 0.7   # medium
        self.w_social = 0.75   # weakest

    def forward(self, x_dict, edge_index_dict):
        user_x, post_x = x_dict['user'], x_dict['post']
        
        # User update: combine three signals
        msg_direct = self.msg_direct(
            (post_x, user_x),
            edge_index_dict[('post', 'rev_engages', 'user')]
        )
        #msg_author = self.msg_author(
         #   (post_x, user_x),
          #  edge_index_dict[('post', 'followed_by', 'user')]
        #)
        msg_social = self.msg_social(
            (user_x, user_x),
            edge_index_dict[('user', 'social', 'user')]
        )
        
        user_out = F.relu(
            self.w_direct * msg_direct +
            #self.w_author * msg_author +
            self.w_social * msg_social
        )
        
        # Post update (optional but helpful)
        post_out = F.relu(
            self.post_update(
                (user_x, post_x),
                edge_index_dict[('user', 'engages', 'post')]
            )
        )
        
        return {'user': user_out, 'post': post_out}

# ----------------------------
# Rest of training code (same as before)
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WeightedRGCN(hidden_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

graph = graph.to(device)
x_dict = {'user': graph['user'].x, 'post': graph['post'].x}
train_edge_index = graph['user', 'engages', 'post'].edge_index

num_users = graph['user'].num_nodes
num_posts = graph['post'].num_nodes


print(f"Training on {train_edge_index.shape[1]} positive edges")
print(f"Users: {num_users}, Posts: {num_posts}")


# Precompute mapping: global post ID → interaction type
global_post_to_interaction = {}
for _, row in train_interactions.iterrows():
    local_post_id = row["post_id"]
    global_post_id = post_to_idx[local_post_id]  # local → global
    global_post_to_interaction[global_post_id] = row["interaction"]

# Convert to tensor for fast lookup (optional but faster)
max_global_post_id = max(post_to_idx.values())
interaction_type_tensor = torch.zeros(max_global_post_id + 1, dtype=torch.float32, device=device)
for gid, inter in global_post_to_interaction.items():
    weight = 3.0 if inter == "QT" else 1.0
    interaction_type_tensor[gid] = weight

# ----------------------------
# Training Function
# ----------------------------
def train():
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(x_dict, graph.edge_index_dict)
    user_emb = out['user']  # [num_users, 64]
    post_emb = out['post']  # [num_posts, 64]

    # Positive edges
    pos_u, pos_p = train_edge_index  # pos_p is LOCAL post index

    # Compute positive scores
    pos_scores = (user_emb[pos_u] * post_emb[pos_p]).sum(dim=1)  # [num_pos]

    # --- Apply weights based on interaction type ---
    # Convert local post index → global post ID
    pos_global_post_ids = pos_p + num_users  # because posts start at index = num_users

    # Fetch weights using precomputed tensor
    pos_weights = interaction_type_tensor[pos_global_post_ids]  # shape: [num_pos]

    # Negative sampling
    neg_p = torch.randint(0, num_posts, (pos_p.size(0),), device=device)
    neg_scores = (user_emb[pos_u] * post_emb[neg_p]).sum(dim=1)

    # Loss with weighting
    pos_loss = criterion(pos_scores, torch.ones_like(pos_scores))
    neg_loss = criterion(neg_scores, torch.zeros_like(neg_scores))

    weighted_pos_loss = (pos_weights * pos_loss).mean()
    loss = weighted_pos_loss + neg_loss

    loss.backward()
    optimizer.step()
    return loss.item()

# ----------------------------
# Evaluation Function (Recall@10, NDCG@10)
# ----------------------------
def evaluate(test_edges, user_emb, post_emb, K=10):
    from collections import defaultdict
    import numpy as np
    from sklearn.metrics import ndcg_score

    # Group test by user
    user_test_posts = defaultdict(list)
    candidate_posts = set()
    
    for i in range(test_edges.shape[1]):
        u = test_edges[0, i].item()
        p_global = test_edges[1, i].item()
        p_local = p_global - num_users
        user_test_posts[u].append(p_local)
        candidate_posts.add(p_local)
    
    candidate_posts = sorted(candidate_posts)
    candidate_posts = torch.tensor(candidate_posts, device=device)
    
    recall_list, ndcg_list = [], []
    
    for user_id in user_test_posts:
        if user_id >= num_users:
            continue
        true_posts = user_test_posts[user_id]
        if not true_posts:
            continue

        # Score all candidates
        scores = torch.mm(
            user_emb[user_id].unsqueeze(0),
            post_emb[candidate_posts].T
        ).squeeze(0)

        # Top-K
        topk_idx = torch.topk(scores, min(K, len(scores)))[1]
        topk_posts = candidate_posts[topk_idx].cpu().tolist()

        # Recall@K
        hits = len(set(topk_posts) & set(true_posts))
        recall = hits / len(true_posts)
        recall_list.append(recall)

        # NDCG@K
        relevance = torch.zeros(len(candidate_posts), device=device)
        for p in true_posts:
            if p in candidate_posts:
                idx = (candidate_posts == p).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    relevance[idx] = 1.0
        if relevance.sum() > 0:
            ndcg = ndcg_score(
                relevance.cpu().numpy().reshape(1, -1),
                scores.cpu().numpy().reshape(1, -1),
                k=K
            )
            ndcg_list.append(ndcg)

    return np.mean(recall_list), np.mean(ndcg_list)

# ----------------------------
# Training Loop
# ----------------------------
print("\n=== START TRAINING ===")
best_recall = 0.0
patience = 20
patience_counter = 0

for epoch in range(1, 201):  # 100 epochs
    loss = train()
    
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            out = model(x_dict, graph.edge_index_dict)
            user_emb = out['user']
            post_emb = out['post']
            
            # Move test edges to device for consistency
            test_edges_device = test_pos_edges.to(device)
            recall, ndcg = evaluate(test_edges_device, user_emb, post_emb, K=10)
            
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Recall@10: {recall:.4f} | NDCG@10: {ndcg:.4f}")
            
            # Early stopping
            if recall > best_recall:
                best_recall = recall
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), "best_rgcn_model.pt")
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

# Save final embeddings
torch.save({
    'user_emb': user_emb.cpu(),
    'post_emb': post_emb.cpu(),
    'user_to_idx': user_to_idx,
    'num_users': num_users,
}, "higgs_embeddings_trained.pt")



# === MANUAL INSPECTION: Print top recommendations for test users ===
print("\n🔍 SAMPLE RECOMMENDATIONS (Top 3 per user):")
#test_users_sample = list(set(test_edges_device[0].cpu().numpy()))[:5]  # First 5 test users
test_users = list(set(test_edges_device[0].cpu().numpy()))

# Randomly sample 5 (or fewer if not enough)
test_users_sample = random.sample(test_users, min(5, len(test_users)))

# Load tweet texts for lookup
activity_sub = data['activity_sub']  # from your loaded .pt file
tweet_lookup = {}
for _, row in activity_sub.iterrows():
    global_post_id = post_to_idx[row["post_id"]]  # local → global
    tweet_lookup[global_post_id] = row["tweet"]

for user_global in test_users_sample:
    user_local = user_global  # users are 0-indexed globally
    true_posts_global = test_edges_device[1][test_edges_device[0] == user_global].cpu().numpy()
    
    # Get candidate posts (all test-period posts)
    candidate_posts_global = set(test_edges_device[1].cpu().numpy())
    candidate_posts_local = [int(gid - num_users) for gid in candidate_posts_global]
    candidate_posts_global = list(candidate_posts_global)
    
    # Score all candidates
    scores = torch.mm(
        user_emb[user_local].unsqueeze(0),
        post_emb[torch.tensor(candidate_posts_local, device=device)].T
    ).squeeze(0)
    
    # Get top-3
    topk_idx = torch.topk(scores, min(3, len(scores)))[1]
    top_posts_local = [candidate_posts_local[i] for i in topk_idx]
    top_posts_global = [pid + num_users for pid in top_posts_local]
    
    print(f"\nUser {user_global}:")
    print("  ✅ True future engagements:")
    for p in true_posts_global:
        tweet = tweet_lookup.get(p, "[MISSING]")
        print(f"    - {tweet}..."+ tweet)
    
    print("  🎯 Top recommendations:")
    for p in top_posts_global:
        tweet = tweet_lookup.get(p, "[MISSING]")
        print(f"    - {tweet}..."+ tweet)
