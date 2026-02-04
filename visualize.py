import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import networkx as nx

# Load your processed data
data = torch.load("synthetic_processed_with_semantics.pt", weights_only=False)
activity_sub = data['activity_sub']
user_to_idx = data['user_to_idx']
post_to_idx = data['post_to_idx']

# Load social graph
social = pd.read_csv("truth_social/follows.tsv", sep="\t", dtype=str)
social.rename(columns={"followed": "followee"}, inplace=True)

print(f"Activity shape: {len(activity_sub)}")

# ----------------------------
# Build a LARGER connected subgraph
# ----------------------------
np.random.seed(42)

# Get users with HIGH activity (more connections = better connectivity)
user_counts = activity_sub["engager"].value_counts()
# Take top 25 most active users (not just 15)
active_users = user_counts.head(25).index

# Also include their target users to ensure connectivity
target_users = set(activity_sub[activity_sub["engager"].isin(active_users)]["target_user"])
all_relevant_users = set(active_users) | target_users

# Get ALL interactions involving these users (not just 30)
sub_activity = activity_sub[
    activity_sub["engager"].isin(all_relevant_users) |
    activity_sub["target_user"].isin(all_relevant_users)
]

# Limit to reasonable size but keep more than 30
if len(sub_activity) > 80:
    sub_activity = sub_activity.head(80)
else:
    sub_activity = sub_activity.copy()

# Get unique users and posts
sub_users = set(sub_activity["engager"]) | set(sub_activity["target_user"])
sub_posts = set(sub_activity["post_id"])

print(f"Initial subgraph: {len(sub_users)} users, {len(sub_posts)} posts, {len(sub_activity)} edges")

# ----------------------------
# Build NetworkX graph
# ----------------------------
G = nx.Graph()

# Add user nodes
for user in sub_users:
    G.add_node(f"u_{user}", node_type="user", label=user)

# Add post nodes  
for post_id in sub_posts:
    post_row = sub_activity[sub_activity["post_id"] == post_id].iloc[0]
    interaction_type = post_row["interaction"]
    G.add_node(f"p_{post_id}", node_type="post", interaction=interaction_type)

# Add engagement edges
for _, row in sub_activity.iterrows():
    engager = f"u_{row['engager']}"
    post = f"p_{row['post_id']}"
    G.add_edge(engager, post, edge_type=row["interaction"])

# Add MORE social edges (up to 25 instead of 10)
social_sub = social[
    social["follower"].isin(sub_users) & 
    social["followee"].isin(sub_users)
].head(25)  # Increased from 10 to 25

for _, row in social_sub.iterrows():
    follower = f"u_{row['follower']}"
    followee = f"u_{row['followee']}"
    if follower in G.nodes and followee in G.nodes:
        G.add_edge(follower, followee, edge_type="social")

# ----------------------------
# Iteratively build largest connected component
# ----------------------------
max_attempts = 10
attempt = 0
target_min_nodes = 50

while attempt < max_attempts and len(G.nodes) < target_min_nodes:
    # If graph is too small, add more users
    if len(G.nodes) == 0:
        break
        
    # Get current connected components
    connected_components = list(nx.connected_components(G))
    if not connected_components:
        break
        
    largest_component = max(connected_components, key=len)
    
    if len(largest_component) >= target_min_nodes:
        break
    
    # Add more users connected to the largest component
    current_users = {node.split('_')[1] for node in largest_component if node.startswith('u_')}
    
    # Find users who interact with current users
    additional_interactions = activity_sub[
        (activity_sub["engager"].isin(current_users)) |
        (activity_sub["target_user"].isin(current_users))
    ]
    
    if len(additional_interactions) == 0:
        break
        
    # Add these interactions
    new_sub_activity = pd.concat([sub_activity, additional_interactions]).drop_duplicates()
    sub_activity = new_sub_activity
    
    # Rebuild graph
    sub_users = set(sub_activity["engager"]) | set(sub_activity["target_user"])
    sub_posts = set(sub_activity["post_id"])
    
    G = nx.Graph()
    for user in sub_users:
        G.add_node(f"u_{user}", node_type="user", label=user)
    for post_id in sub_posts:
        post_row = sub_activity[sub_activity["post_id"] == post_id].iloc[0]
        interaction_type = post_row["interaction"]
        G.add_node(f"p_{post_id}", node_type="post", interaction=interaction_type)
    for _, row in sub_activity.iterrows():
        engager = f"u_{row['engager']}"
        post = f"p_{row['post_id']}"
        G.add_edge(engager, post, edge_type=row["interaction"])
    social_sub = social[
        social["follower"].isin(sub_users) & 
        social["followee"].isin(sub_users)
    ].head(30)
    for _, row in social_sub.iterrows():
        follower = f"u_{row['follower']}"
        followee = f"u_{row['followee']}"
        if follower in G.nodes and followee in G.nodes:
            G.add_edge(follower, followee, edge_type="social")
    
    attempt += 1

# Final connected component extraction
connected_components = list(nx.connected_components(G))
if connected_components:
    largest_component = max(connected_components, key=len)
    G_connected = G.subgraph(largest_component).copy()
    print(f"Final connected component: {len(G_connected.nodes)} nodes, {len(G_connected.edges)} edges")
    G = G_connected
else:
    print("No connected components found")
    exit()

# Ensure we have enough nodes
if len(G.nodes) < 20:
    print("Warning: Graph still too small, but proceeding anyway")

# ----------------------------
# Create layout with semantic positioning
# ----------------------------
post_texts = {}
post_nodes = [node for node in G.nodes if G.nodes[node]["node_type"] == "post"]
for node in post_nodes:
    post_id = int(node.split('_')[1])
    post_row = sub_activity[sub_activity["post_id"] == post_id]
    if len(post_row) > 0:
        post_texts[node] = post_row.iloc[0]["tweet"][:50] + "..."
    else:
        post_texts[node] = "[MISSING]"

# Encode post texts
if post_nodes:
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    post_embeddings = []
    for node in post_nodes:
        text = post_texts.get(node, "[EMPTY]")
        emb = encoder.encode([text])[0]
        post_embeddings.append(emb)
    
    post_embeddings = np.array(post_embeddings)
    
    # Reduce to 2D for layout
    pca = PCA(n_components=2)
    post_coords = pca.fit_transform(post_embeddings)
    
    pos = {}
    for i, node in enumerate(post_nodes):
        pos[node] = post_coords[i]
    
    user_nodes = [node for node in G.nodes if G.nodes[node]["node_type"] == "user"]
    for node in user_nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            neighbor_positions = [pos[n] for n in neighbors if n in pos]
            if neighbor_positions:
                pos[node] = np.mean(neighbor_positions, axis=0) + np.random.normal(0, 0.1, 2)
            else:
                pos[node] = np.random.rand(2)
        else:
            pos[node] = np.random.rand(2)
else:
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)

# ----------------------------
# Plot the larger connected graph
# ----------------------------
plt.figure(figsize=(20, 15))

user_nodes = [node for node in G.nodes if G.nodes[node]["node_type"] == "user"]
post_nodes = [node for node in G.nodes if G.nodes[node]["node_type"] == "post"]

# User colors by degree
user_degrees = [G.degree(node) for node in user_nodes]
user_colors = plt.cm.viridis(np.array(user_degrees) / max(user_degrees) if user_degrees else 1)
user_colors = "#45B7D1"
# Post colors by interaction type
post_colors = []
for node in post_nodes:
    interaction = G.nodes[node]["interaction"]
    if interaction == "QT":
        post_colors.append("#FF6B6B")  # Red for quotes
    elif interaction == "RE":
        post_colors.append("#4ECDC4")  # Teal for replies  
    else:  # POST
        post_colors.append("#45B7D1")  # Blue for posts

# Draw nodes with larger sizes for better visibility
nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, 
                      node_color=user_colors, node_size=1000, 
                      node_shape='o', alpha=0.8, edgecolors='black', linewidths=1.5)

nx.draw_networkx_nodes(G, pos, nodelist=post_nodes, 
                      node_color=post_colors, node_size=800, 
                      node_shape='s', alpha=0.8, edgecolors='black', linewidths=1.5)

# Draw edges
edge_types = nx.get_edge_attributes(G, 'edge_type')
qt_edges = [edge for edge, etype in edge_types.items() if etype == "QT"]
re_edges = [edge for edge, etype in edge_types.items() if etype == "RE"]
post_edges = [edge for edge, etype in edge_types.items() if etype == "POST"]
social_edges = [edge for edge, etype in edge_types.items() if etype == "social"]

nx.draw_networkx_edges(G, pos, edgelist=qt_edges, 
                      edge_color='#FF6B6B', width=2.5, alpha=0.7)
nx.draw_networkx_edges(G, pos, edgelist=re_edges, 
                      edge_color='#4ECDC4', width=2.5, alpha=0.7)
nx.draw_networkx_edges(G, pos, edgelist=post_edges, 
                      edge_color='#45B7D1', width=2.5, alpha=0.7)
nx.draw_networkx_edges(G, pos, edgelist=social_edges, 
                      edge_color='gray', width=1.5, alpha=0.5, style='dashed')

# Add labels (only for larger graphs)
if len(G.nodes) <= 60:
    user_labels = {node: f"U{node.split('_')[1][:4]}..." for node in user_nodes}
    post_labels = {node: f"P{node.split('_')[1][:3]}" for node in post_nodes}
    all_labels = {**user_labels, **post_labels}
    nx.draw_networkx_labels(G, pos, labels=all_labels, font_size=9, font_weight='bold')

# Create legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=8, label='User'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF6B6B', 
               markersize=8, label='Post'),
    plt.Line2D([0], [0], color='#FF6B6B', linewidth=2.5, label='Engagement Edge'),
    plt.Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--', label='Social Edge')
]

plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=12)

plt.title(f"Heterogeneous Social Recommendation Graph\n({len(G.nodes)} Nodes, {len(G.edges)} Edges - Fully Connected)", 
          fontsize=18, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()

# Save high-quality image for PPT
plt.savefig("truth_social_graph_ppt_large.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print(f"\n✅ Final visualization ready: {len(G.nodes)} nodes, {len(G.edges)} edges")