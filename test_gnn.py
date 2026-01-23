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
data = torch.load("synthetic_processed_with_semantics.pt", weights_only=False)
activity_sub = data['activity_sub']
user_to_idx = data['user_to_idx']
post_to_idx = data['post_to_idx']
x = data['x']

num_users = data['num_users']
num_posts = data['num_posts']

# Temporal split (same as training)
activity_sorted = activity_sub.sort_values("timestamp").reset_index(drop=True)
n = len(activity_sorted)
train_end = int(0.8 * n)
val_end = int(0.9 * n)
test_interactions = activity_sorted.iloc[val_end:]

# Build test edges
def build_test_edges(df, user_to_idx, post_to_idx):
    engager, post_global = [], []
    for _, row in df.iterrows():
        u_eng = user_to_idx.get(row["engager"])
        p_global = post_to_idx.get(row["post_id"])
        if u_eng is not None and p_global is not None:
            engager.append(u_eng)
            post_global.append(p_global)
    return torch.tensor([engager, post_global], dtype=torch.long)

test_pos_edges = build_test_edges(test_interactions, user_to_idx, post_to_idx)

# Find top 10 most active users in test set
user_test_counts = defaultdict(int)
for i in range(test_pos_edges.shape[1]):
    user_id = test_pos_edges[0, i].item()
    user_test_counts[user_id] += 1

# Get top 10 users by test engagement count
top_10_users = sorted(user_test_counts.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"Top 10 most active test users:")
for i, (user_id, count) in enumerate(top_10_users):
    print(f"  {i+1}. User {user_id}: {count} engagements")

# ----------------------------
# 2. Rebuild graph for inference
# ----------------------------
graph = HeteroData()
graph['user'].x = x[:num_users]
graph['post'].x = x[num_users:]

# Add edges (same as training)
src, dst = data['edge_index_social']
graph['user', 'social', 'user'].edge_index = torch.stack([src, dst], dim=0)

src, dst = data['edge_index_engage']
post_local = dst - num_users
mask = (src < num_users) & (post_local >= 0) & (post_local < num_posts)
graph['user', 'engages', 'post'].edge_index = torch.stack([src[mask], post_local[mask]], dim=0)

src, dst = data['edge_index_author']
post_local = src - num_users
mask = (post_local >= 0) & (post_local < num_posts) & (dst < num_users)
graph['post', 'authored_by', 'user'].edge_index = torch.stack([post_local[mask], dst[mask]], dim=0)

src, dst = data['edge_index_followed_by']
if src.numel() > 0:
    post_local = src - num_users
    mask = (post_local >= 0) & (post_local < num_posts) & (dst < num_users)
    graph['post', 'followed_by', 'user'].edge_index = torch.stack([post_local[mask], dst[mask]], dim=0)
else:
    graph['post', 'followed_by', 'user'].edge_index = torch.empty((2, 0), dtype=torch.long)

graph['post', 'rev_engages', 'user'].edge_index = graph['user', 'engages', 'post'].edge_index.flip(0)

# ----------------------------
# 3. Load trained model
# ----------------------------
class WeightedRGCN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.msg_direct = SAGEConv((-1, -1), hidden_dim)   # user ← post (engagement)
        self.msg_author = SAGEConv((-1, -1), hidden_dim)   # user ← post (follows author)
        self.msg_social = SAGEConv((-1, -1), hidden_dim)   # user ← user (social)
        self.post_update = SAGEConv((-1, -1), hidden_dim)  # post ← user (engagement)
        
        # FIXED weights
        self.w_direct = 1.75   # strongest
        self.w_author = 0.7   # medium
        self.w_social = 0.3   # weakest

    def forward(self, x_dict, edge_index_dict):
        user_x, post_x = x_dict['user'], x_dict['post']
        
        # User update: combine three signals
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
        
        user_out = F.relu(
            self.w_direct * msg_direct +
            self.w_author * msg_author +
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WeightedRGCN(hidden_dim=64).to(device)
model.load_state_dict(torch.load("best_rgcn_model.pt", weights_only=True))
model.eval()

graph = graph.to(device)
x_dict = {'user': graph['user'].x, 'post': graph['post'].x}

# ----------------------------
# 4. Compute embeddings
# ----------------------------
with torch.no_grad():
    out = model(x_dict, graph.edge_index_dict)
    user_emb = out['user']
    post_emb = out['post']

# ----------------------------
# 5. Evaluate TOP 10 MOST ACTIVE USERS
# ----------------------------
test_edges_device = test_pos_edges.to(device)

# Candidate posts = all test posts
candidate_posts_global = set(test_edges_device[1].cpu().numpy())
candidate_posts_local = sorted([int(p - num_users) for p in candidate_posts_global])
candidate_posts_local = torch.tensor(candidate_posts_local, device=device)

# Tweet lookup
tweet_lookup = {}
for _, row in activity_sub.iterrows():
    global_id = post_to_idx[row["post_id"]]
    tweet_lookup[global_id] = row["tweet"]

# Store results
results = []

print("\n" + "="*60)
print("EVALUATION FOR TOP 10 MOST ACTIVE TEST USERS")
print("="*60)

for rank, (user_id, true_count) in enumerate(top_10_users, 1):
    # Get true test posts for this user
    true_mask = (test_edges_device[0] == user_id)
    true_posts_global = test_edges_device[1][true_mask].cpu().numpy()
    true_posts_local = [int(p - num_users) for p in true_posts_global]
    
    if not true_posts_local:
        continue
        
    # Score candidates
    scores = torch.mm(
        user_emb[user_id].unsqueeze(0),
        post_emb[candidate_posts_local].T
    ).squeeze(0)
    
    # Top-10
    K = 10
    topk_idx = torch.topk(scores, min(K, len(scores)))[1]
    topk_posts_local = candidate_posts_local[topk_idx].cpu().tolist()
    
    # Recall@10
    hits = len(set(topk_posts_local) & set(true_posts_local))
    recall_at_10 = hits / len(true_posts_local)
    
    # NDCG@10
    relevance = torch.zeros(len(candidate_posts_local), device=device)
    for p in true_posts_local:
        if p in candidate_posts_local:
            idx = (candidate_posts_local == p).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                relevance[idx] = 1.0
    
    ndcg_at_10 = 0.0
    if relevance.sum() > 0:
        ndcg_at_10 = ndcg_score(
            relevance.cpu().numpy().reshape(1, -1),
            scores.cpu().numpy().reshape(1, -1),
            k=K
        )
    
    results.append((user_id, true_count, recall_at_10, ndcg_at_10))
    
    print(f"\n{rank}. USER {user_id} (Test Engagements: {true_count})")
    print(f"   Recall@10: {recall_at_10:.4f} | NDCG@10: {ndcg_at_10:.4f}")
    
    # Show samples only for first 3 users to avoid clutter
    if rank <= 3:
        print(f"   ✅ True engagements (first 3):")
        for p_global in true_posts_global[:3]:
            tweet = tweet_lookup.get(p_global, "[MISSING]")
            print(f"     - {tweet[:60]}...")
        
        print(f"   🎯 Top recommendations (first 3):")
        top_posts_global = [int(p + num_users) for p in topk_posts_local[:3]]
        for p_global in top_posts_global:
            tweet = tweet_lookup.get(p_global, "[MISSING]")
            print(f"     - {tweet[:60]}...")

# ----------------------------
# 6. Summary Statistics
# ----------------------------
recall_scores = [r[2] for r in results]
ndcg_scores = [r[3] for r in results]

print("\n" + "="*60)
print("SUMMARY FOR TOP 10 MOST ACTIVE USERS")
print("="*60)
print(f"Average Recall@10: {np.mean(recall_scores):.4f}")
print(f"Average NDCG@10:  {np.mean(ndcg_scores):.4f}")
print(f"Users with Recall@10 > 0: {sum(1 for r in recall_scores if r > 0)} / 10")