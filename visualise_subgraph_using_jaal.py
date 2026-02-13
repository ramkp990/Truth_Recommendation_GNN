import pandas as pd
import torch
from jaal import Jaal
import random

# ----------------------------
# 1. Load processed graph data
# ----------------------------
print("Loading graph data...")
# Add safe globals for loading the pickle file
torch.serialization.add_safe_globals([pd.DataFrame])
data = torch.load("synthetic_processed_with_semantics.pt", weights_only=False)

user_to_idx = data['user_to_idx']
post_to_idx = data['post_to_idx']
edge_index_social = data['edge_index_social']
edge_index_engage = data['edge_index_engage']
edge_index_author = data['edge_index_author']
edge_index_followed_by = data['edge_index_followed_by']
num_users = data['num_users']
num_posts = data['num_posts']

# Reverse mappings
idx_to_user = {idx: user for user, idx in user_to_idx.items()}
idx_to_post = {idx: post_id for post_id, idx in post_to_idx.items()}

print(f"Loaded graph with {num_users} users and {num_posts} posts")

# ----------------------------
# 2. Pick one user with multiple edge types
# ----------------------------
# Find a user who has multiple types of interactions
def find_good_user():
    """Find a user with various edge types for rich visualization"""
    user_scores = {}
    
    # Count outgoing social edges (follows)
    for i in range(edge_index_social.shape[1]):
        src = edge_index_social[0, i].item()
        if src < num_users:
            user_scores[src] = user_scores.get(src, 0) + 1
    
    # Count engagements
    for i in range(edge_index_engage.shape[1]):
        src = edge_index_engage[0, i].item()
        if src < num_users:
            user_scores[src] = user_scores.get(src, 0) + 2  # Weight engagements higher
    
    # Sort by activity
    sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Pick from top 10 most active users
    if sorted_users:
        top_users = sorted_users[:min(10, len(sorted_users))]
        selected_idx = random.choice(top_users)[0]
        return selected_idx
    return 0

selected_user_idx = find_good_user()
selected_user_id = idx_to_user[selected_user_idx]

print(f"\n🎯 Selected User: {selected_user_id} (index: {selected_user_idx})")

# ----------------------------
# 3. Extract edges for the selected user
# ----------------------------
edges_data = []
nodes_set = {selected_user_idx}  # Start with the selected user

# === SOCIAL EDGES (follows) ===
# User follows other users
for i in range(edge_index_social.shape[1]):
    src, dst = edge_index_social[0, i].item(), edge_index_social[1, i].item()
    if src == selected_user_idx and dst < num_users:
        edges_data.append({
            'from': f"U{selected_user_id}",
            'to': f"U{idx_to_user[dst]}",
            'type': 'follows',
            'color': '#3498db'  # Blue
        })
        nodes_set.add(dst)

# Users who follow the selected user
for i in range(edge_index_social.shape[1]):
    src, dst = edge_index_social[0, i].item(), edge_index_social[1, i].item()
    if dst == selected_user_idx and src < num_users:
        edges_data.append({
            'from': f"U{idx_to_user[src]}",
            'to': f"U{selected_user_id}",
            'type': 'is_followed_by',
            'color': '#9b59b6'  # Purple
        })
        nodes_set.add(src)

# === ENGAGEMENT EDGES ===
# User engages with posts
for i in range(edge_index_engage.shape[1]):
    src, dst = edge_index_engage[0, i].item(), edge_index_engage[1, i].item()
    if src == selected_user_idx:
        post_local_id = idx_to_post.get(dst, dst - num_users)
        edges_data.append({
            'from': f"U{selected_user_id}",
            'to': f"P{post_local_id}",
            'type': 'engages_with',
            'color': '#e74c3c'  # Red
        })
        nodes_set.add(dst)

# === AUTHORSHIP EDGES ===
# Posts authored by the user
for i in range(edge_index_author.shape[1]):
    src, dst = edge_index_author[0, i].item(), edge_index_author[1, i].item()
    if dst == selected_user_idx:
        post_local_id = idx_to_post.get(src, src - num_users)
        edges_data.append({
            'from': f"P{post_local_id}",
            'to': f"U{selected_user_id}",
            'type': 'authored_by',
            'color': '#2ecc71'  # Green
        })
        nodes_set.add(src)

# Users who engage with selected user's posts
for i in range(edge_index_author.shape[1]):
    src_post, dst_author = edge_index_author[0, i].item(), edge_index_author[1, i].item()
    if dst_author == selected_user_idx:
        # Find who engaged with this post
        for j in range(edge_index_engage.shape[1]):
            eng_src, eng_dst = edge_index_engage[0, j].item(), edge_index_engage[1, j].item()
            if eng_dst == src_post and eng_src != selected_user_idx and eng_src < num_users:
                post_local_id = idx_to_post.get(src_post, src_post - num_users)
                edges_data.append({
                    'from': f"U{idx_to_user[eng_src]}",
                    'to': f"P{post_local_id}",
                    'type': 'others_engage',
                    'color': '#f39c12'  # Orange
                })
                nodes_set.add(eng_src)
                nodes_set.add(src_post)

# === FOLLOWED_BY EDGES ===
# Posts that reach the user through follows
for i in range(edge_index_followed_by.shape[1]):
    src_post, dst_user = edge_index_followed_by[0, i].item(), edge_index_followed_by[1, i].item()
    if dst_user == selected_user_idx:
        post_local_id = idx_to_post.get(src_post, src_post - num_users)
        edges_data.append({
            'from': f"P{post_local_id}",
            'to': f"U{selected_user_id}",
            'type': 'reaches_via_follow',
            'color': '#1abc9c'  # Teal
        })
        nodes_set.add(src_post)

print(f"\n📊 Graph Statistics:")
print(f"   Total nodes: {len(nodes_set)}")
print(f"   Total edges: {len(edges_data)}")

# ----------------------------
# 4. Create node DataFrame
# ----------------------------
nodes_data = []
for node_idx in nodes_set:
    if node_idx < num_users:
        # User node
        user_id = idx_to_user[node_idx]
        node_type = "Selected User" if node_idx == selected_user_idx else "User"
        nodes_data.append({
            'id': f"U{user_id}",
            'type': node_type,
            'label': f"U{user_id}",
            'color': '#e74c3c' if node_idx == selected_user_idx else '#3498db'
        })
    else:
        # Post node
        post_local_id = idx_to_post.get(node_idx, node_idx - num_users)
        nodes_data.append({
            'id': f"P{post_local_id}",
            'type': 'Post',
            'label': f"P{post_local_id}",
            'color': '#95a5a6'  # Gray
        })

edge_df = pd.DataFrame(edges_data)
node_df = pd.DataFrame(nodes_data)

# ----------------------------
# 5. Display edge type statistics
# ----------------------------
print(f"\n📈 Edge Type Distribution:")
edge_counts = edge_df['type'].value_counts()
for edge_type, count in edge_counts.items():
    print(f"   {edge_type}: {count}")

# ----------------------------
# 6. Visualize with Jaal
# ----------------------------
print(f"\n🚀 Launching Jaal visualization...")
print(f"\n🎨 Legend:")
print(f"   🔵 Blue edges: follows (user follows other users)")
print(f"   🟣 Purple edges: is_followed_by (other users follow selected user)")
print(f"   🔴 Red edges: engages_with (user engages with posts)")
print(f"   🟢 Green edges: authored_by (posts authored by user)")
print(f"   🟠 Orange edges: others_engage (others engage with user's posts)")
print(f"   🟢 Teal edges: reaches_via_follow (posts reach user via follows)")
print(f"\n   🔴 Red node: Selected user")
print(f"   🔵 Blue nodes: Other users")
print(f"   ⚫ Gray nodes: Posts")

# Launch Jaal
Jaal(edge_df, node_df).plot()
