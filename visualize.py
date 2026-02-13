# Creates a semantically-aware visualization of a connected heterogeneous social network graph with users and posts as nodes, colored by interaction type and positioned using text embeddings for meaningful spatial arrangement.

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import networkx as nx

# Load preprocessed graph data containing activity records and node mappings
data = torch.load("synthetic_processed_with_semantics.pt", weights_only=False)
activity_sub = data['activity_sub']        # Activity dataframe with engagements
user_to_idx = data['user_to_idx']          # User ID → node index mapping
post_to_idx = data['post_to_idx']          # Post ID → node index mapping

# Load social graph (follow relationships)
social = pd.read_csv("truth_social/follows.tsv", sep="\t", dtype=str)
social.rename(columns={"followed": "followee"}, inplace=True)  # Standardize column name

print(f"Activity shape: {len(activity_sub)}")

# ----------------------------
# Build a LARGER connected subgraph for visualization
# ----------------------------
np.random.seed(42)  # For reproducible sampling

# Identify highly active users (top 25 by engagement count) to ensure connectivity
user_counts = activity_sub["engager"].value_counts()
active_users = user_counts.head(25).index  # Increased from 15 to 25 for richer graph

# Include target users (those being engaged with) to maintain interaction connectivity
target_users = set(activity_sub[activity_sub["engager"].isin(active_users)]["target_user"])
all_relevant_users = set(active_users) | target_users  # Union of engagers and targets

# Extract ALL interactions involving relevant users (not limited to 30)
sub_activity = activity_sub[
    activity_sub["engager"].isin(all_relevant_users) |
    activity_sub["target_user"].isin(all_relevant_users)
]

# Cap at 80 interactions to maintain readability while ensuring sufficient connectivity
if len(sub_activity) > 80:
    sub_activity = sub_activity.head(80)
else:
    sub_activity = sub_activity.copy()

# Extract unique users and posts from filtered activity
sub_users = set(sub_activity["engager"]) | set(sub_activity["target_user"])
sub_posts = set(sub_activity["post_id"])

print(f"Initial subgraph: {len(sub_users)} users, {len(sub_posts)} posts, {len(sub_activity)} edges")

# ----------------------------
# Build NetworkX graph with heterogeneous nodes and edges
# ----------------------------
G = nx.Graph()  # Undirected graph for visualization (directionality not needed for layout)

# Add user nodes with metadata
for user in sub_users:
    G.add_node(f"u_{user}", node_type="user", label=user)  # Prefix 'u_' distinguishes user nodes

# Add post nodes with interaction type metadata
for post_id in sub_posts:
    post_row = sub_activity[sub_activity["post_id"] == post_id].iloc[0]
    interaction_type = post_row["interaction"]  # QT/RE/POST
    G.add_node(f"p_{post_id}", node_type="post", interaction=interaction_type)  # Prefix 'p_' for posts

# Add engagement edges (user → post) with interaction type labels
for _, row in sub_activity.iterrows():
    engager = f"u_{row['engager']}"
    post = f"p_{row['post_id']}"
    G.add_edge(engager, post, edge_type=row["interaction"])  # Edge labeled with interaction type

# Add social edges (user ↔ user follow relationships) - increased from 10 to 25 for better connectivity
social_sub = social[
    social["follower"].isin(sub_users) & 
    social["followee"].isin(sub_users)
].head(25)  # Limit to 25 edges to avoid clutter while improving connectivity

for _, row in social_sub.iterrows():
    follower = f"u_{row['follower']}"
    followee = f"u_{row['followee']}"
    # Only add edge if both users exist in our subgraph
    if follower in G.nodes and followee in G.nodes:
        G.add_edge(follower, followee, edge_type="social")  # Undirected social edge

# ----------------------------
# Iteratively build largest connected component (prevent fragmented visualization)
# ----------------------------
max_attempts = 10      # Maximum expansion attempts
attempt = 0
target_min_nodes = 50  # Target size for connected component

# Iteratively expand graph until we reach target size or exhaust attempts
while attempt < max_attempts and len(G.nodes) < target_min_nodes:
    # Skip if graph is empty (shouldn't happen with our initialization)
    if len(G.nodes) == 0:
        break
        
    # Identify largest connected component
    connected_components = list(nx.connected_components(G))
    if not connected_components:
        break
        
    largest_component = max(connected_components, key=len)
    
    # Stop if we've reached target size
    if len(largest_component) >= target_min_nodes:
        break
    
    # Expand by adding users connected to current largest component
    current_users = {node.split('_')[1] for node in largest_component if node.startswith('u_')}
    
    # Find additional interactions involving current users
    additional_interactions = activity_sub[
        (activity_sub["engager"].isin(current_users)) |
        (activity_sub["target_user"].isin(current_users))
    ]
    
    # Stop if no new interactions found
    if len(additional_interactions) == 0:
        break
        
    # Merge new interactions into subgraph
    new_sub_activity = pd.concat([sub_activity, additional_interactions]).drop_duplicates()
    sub_activity = new_sub_activity
    
    # Rebuild graph with expanded activity set
    sub_users = set(sub_activity["engager"]) | set(sub_activity["target_user"])
    sub_posts = set(sub_activity["post_id"])
    
    # Reconstruct graph from scratch with expanded nodes/edges
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
    # Add more social edges (30 instead of 25) for better connectivity
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

# Extract largest connected component for final visualization
connected_components = list(nx.connected_components(G))
if connected_components:
    largest_component = max(connected_components, key=len)
    G_connected = G.subgraph(largest_component).copy()  # Create subgraph view of largest component
    print(f"Final connected component: {len(G_connected.nodes)} nodes, {len(G_connected.edges)} edges")
    G = G_connected
else:
    print("No connected components found")
    exit()

# Safety check: proceed even with smaller graphs (but warn user)
if len(G.nodes) < 20:
    print("Warning: Graph still too small, but proceeding anyway")

# ----------------------------
# Create layout with semantic positioning using post text embeddings
# ----------------------------
# Build lookup dictionary for post texts (truncated for labeling)
post_texts = {}
post_nodes = [node for node in G.nodes if G.nodes[node]["node_type"] == "post"]
for node in post_nodes:
    post_id = int(node.split('_')[1])
    post_row = sub_activity[sub_activity["post_id"] == post_id]
    if len(post_row) > 0:
        post_texts[node] = post_row.iloc[0]["tweet"][:50] + "..."  # Truncate to 50 chars
    else:
        post_texts[node] = "[MISSING]"

# Generate semantic embeddings for posts to position similar content nearby
if post_nodes:
    encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight sentence encoder
    post_embeddings = []
    for node in post_nodes:
        text = post_texts.get(node, "[EMPTY]")
        emb = encoder.encode([text])[0]  # Generate 384-dim embedding
        post_embeddings.append(emb)
    
    post_embeddings = np.array(post_embeddings)
    
    # Reduce dimensionality to 2D for spatial layout using PCA
    pca = PCA(n_components=2)
    post_coords = pca.fit_transform(post_embeddings)  # [num_posts, 2]
    
    # Initialize position dictionary with semantic post positions
    pos = {}
    for i, node in enumerate(post_nodes):
        pos[node] = post_coords[i]
    
    # Position users near their connected posts (centroid of neighbors + noise)
    user_nodes = [node for node in G.nodes if G.nodes[node]["node_type"] == "user"]
    for node in user_nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            # Average position of connected posts
            neighbor_positions = [pos[n] for n in neighbors if n in pos]
            if neighbor_positions:
                pos[node] = np.mean(neighbor_positions, axis=0) + np.random.normal(0, 0.1, 2)
            else:
                pos[node] = np.random.rand(2)  # Fallback random position
        else:
            pos[node] = np.random.rand(2)  # Isolated nodes get random position
else:
    # Fallback to force-directed layout if no posts available
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)

# ----------------------------
# Plot the larger connected heterogeneous graph
# ----------------------------
plt.figure(figsize=(20, 15))  # Large canvas for detailed visualization

# Separate nodes by type for styling
user_nodes = [node for node in G.nodes if G.nodes[node]["node_type"] == "user"]
post_nodes = [node for node in G.nodes if G.nodes[node]["node_type"] == "post"]

# Style user nodes: uniform color with degree-based sizing (commented out degree coloring)
user_degrees = [G.degree(node) for node in user_nodes]
user_colors = plt.cm.viridis(np.array(user_degrees) / max(user_degrees) if user_degrees else 1)
user_colors = "#45B7D1"  # Uniform blue color for users (overriding degree-based coloring)

# Style post nodes by interaction type with distinct colors
post_colors = []
for node in post_nodes:
    interaction = G.nodes[node]["interaction"]
    if interaction == "QT":
        post_colors.append("#FF6B6B")  # Coral red for quotes (high semantic value)
    elif interaction == "RE":
        post_colors.append("#4ECDC4")  # Teal for replies
    else:  # POST
        post_colors.append("#45B7D1")  # Blue for original posts

# Draw user nodes (circles) with enhanced visibility
nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, 
                      node_color=user_colors, node_size=1000, 
                      node_shape='o', alpha=0.8, edgecolors='black', linewidths=1.5)

# Draw post nodes (squares) with interaction-type coloring
nx.draw_networkx_nodes(G, pos, nodelist=post_nodes, 
                      node_color=post_colors, node_size=800, 
                      node_shape='s', alpha=0.8, edgecolors='black', linewidths=1.5)

# Separate edges by type for distinct styling
edge_types = nx.get_edge_attributes(G, 'edge_type')
qt_edges = [edge for edge, etype in edge_types.items() if etype == "QT"]      # Quote edges
re_edges = [edge for edge, etype in edge_types.items() if etype == "RE"]      # Reply edges
post_edges = [edge for edge, etype in edge_types.items() if etype == "POST"]  # Post edges
social_edges = [edge for edge, etype in edge_types.items() if etype == "social"]  # Social edges

# Draw edges with type-specific styling
nx.draw_networkx_edges(G, pos, edgelist=qt_edges, 
                      edge_color='#FF6B6B', width=2.5, alpha=0.7)  # Bold red for quotes
nx.draw_networkx_edges(G, pos, edgelist=re_edges, 
                      edge_color='#4ECDC4', width=2.5, alpha=0.7)  # Teal for replies
nx.draw_networkx_edges(G, pos, edgelist=post_edges, 
                      edge_color='#45B7D1', width=2.5, alpha=0.7)  # Blue for posts
nx.draw_networkx_edges(G, pos, edgelist=social_edges, 
                      edge_color='gray', width=1.5, alpha=0.5, style='dashed')  # Dashed gray for social

# Add abbreviated labels for readability (only if graph isn't too dense)
if len(G.nodes) <= 60:
    user_labels = {node: f"U{node.split('_')[1][:4]}..." for node in user_nodes}  # "U<first4chars>..."
    post_labels = {node: f"P{node.split('_')[1][:3]}" for node in post_nodes}     # "P<post_id>"
    all_labels = {**user_labels, **post_labels}
    nx.draw_networkx_labels(G, pos, labels=all_labels, font_size=9, font_weight='bold')

# Create legend for graph elements
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=8, label='User'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF6B6B', 
               markersize=8, label='Post'),
    plt.Line2D([0], [0], color='#FF6B6B', linewidth=2.5, label='Engagement Edge'),
    plt.Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--', label='Social Edge')
]

plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=12)

# Add descriptive title with graph statistics
plt.title(f"Heterogeneous Social Recommendation Graph\n({len(G.nodes)} Nodes, {len(G.edges)} Edges - Fully Connected)", 
          fontsize=18, fontweight='bold', pad=20)
plt.axis('off')  # Hide axes for cleaner visualization
plt.tight_layout()

# Save high-resolution image suitable for presentations/publications
plt.savefig("truth_social_graph_ppt_large.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print(f"\n✅ Final visualization ready: {len(G.nodes)} nodes, {len(G.edges)} edges")
