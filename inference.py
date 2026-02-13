# Evaluates an inductive graph neural network for recommending social media posts to both seen and unseen users by generating structural embeddings and computing similarity with post embeddings.

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import re
import unicodedata
from urllib.parse import urlparse
from torch_geometric.data import HeteroData

# ----------------------------
# Text Cleaning (same as training)
# ----------------------------
def extract_url_features(text):
    """
    Extract and enrich URL information from text by normalizing malformed URLs,
    parsing domain/path components, and generating descriptive keywords from URL paths.
    
    Args:
        text (str): Raw tweet text potentially containing URLs
    
    Returns:
        str: Original text if no URLs found, or enriched URL representation like "[LINK] domain keyword1 keyword2..."
    """
    # Normalize malformed URLs (e.g., "https : //example.com" → "https://example.com")
    text = re.sub(r'https?\s*:\s*//', 'https://', text, flags=re.IGNORECASE)
    
    # Extract valid URLs using regex pattern
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    
    # Return original text if no URLs found
    if not urls:
        return text
    
    # Process only the first URL found in the text
    main_url = urls[0]
    parsed = urlparse(main_url)
    domain = parsed.netloc.lower()  # Extract and normalize domain (e.g., "EXAMPLE.COM" → "example.com")
    path = parsed.path.lower()      # Extract and normalize path component
    
    # Extract meaningful keywords from URL path (split by delimiters, filter short/non-alphabetic words)
    keywords = [word for word in re.split(r'[-_/]', path) if word.isalpha() and len(word) > 2]
    keyword_str = " ".join(keywords[:5])  # Limit to first 5 keywords
    
    # Return enriched representation with domain and keywords if available, otherwise just domain
    if keyword_str:
        return f"[LINK] {domain} {keyword_str}"
    else:
        return f"[LINK] {domain}"

def clean_tweet_text(text):
    """
    Clean and normalize tweet text with special handling for URL-heavy posts, emojis, and repetitive content.
    
    Args:
        text (str): Raw tweet text
    
    Returns:
        str: Cleaned text or special tokens like "[EMPTY]" or enriched URL representation
    """
    # Handle non-string or empty inputs
    if not isinstance(text, str) or not text.strip():
        return "[EMPTY]"
    
    # Count URLs and words to detect URL-spam posts (≥1 URL and ≤5 words)
    url_count = len(re.findall(r'https?://', text))
    word_count = len(re.findall(r'\b\w+\b', text))
    
    # For URL-spam posts, replace with enriched URL representation instead of full text
    if url_count >= 1 and word_count <= 5:
        return extract_url_features(text)
    
    # Normalize Unicode characters (e.g., compatibility characters → standard forms)
    text = unicodedata.normalize('NFKC', text)
    
    # Collapse multiple whitespace characters into single spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Replace emoji placeholders with standardized token
    text = re.sub(r'<emoji:\s*\w+\s*>', ' [EMOJI] ', text)
    
    # Remove duplicate consecutive sentences (split by periods)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    unique_sentences = []
    for s in sentences:
        if not unique_sentences or s != unique_sentences[-1]:  # Skip if identical to previous sentence
            unique_sentences.append(s)
    text = '. '.join(unique_sentences) + ('.' if unique_sequences else '')
    
    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle very short/empty results after cleaning
    if not text or len(text) < 3:
        return "[EMPTY]"
    
    # Truncate extremely long texts to 512 characters
    return text[:512]

# ----------------------------
# Load trained model and data
# ----------------------------
# Set computation device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load preprocessed graph data containing node features and structural information
data = torch.load("synthetic_processed_with_semantics.pt", weights_only=False)
x = data['x']  # Combined node feature matrix [users; posts]
num_users = data['num_users']  # Number of user nodes
num_posts = data['num_posts']  # Number of post nodes

# Define WeightedRGCN model architecture matching training configuration
from torch_geometric.nn import SAGEConv

class WeightedRGCN(torch.nn.Module):
    """
    Weighted Relational Graph Convolutional Network for heterogeneous social media graphs.
    Combines direct engagement messages with social graph messages using learned weights.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Message passing from posts to users (reverse engagement edges)
        self.msg_direct = SAGEConv((-1, -1), hidden_dim)
        # Message passing along social graph edges (user-to-user)
        self.msg_social = SAGEConv((-1, -1), hidden_dim)
        # Message passing from users to posts (engagement edges)
        self.post_update = SAGEConv((-1, -1), hidden_dim)
        # Fixed weights for combining message types (empirically tuned)
        self.w_direct = 1.0
        self.w_social = 0.75

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through heterogeneous graph:
        1. Aggregate messages from posts users engaged with (direct signal)
        2. Aggregate messages from social connections (social signal)
        3. Combine signals with weighted sum
        4. Update post representations using user engagements
        """
        user_x, post_x = x_dict['user'], x_dict['post']
        
        # Direct engagement messages: posts → users via reverse engagement edges
        msg_direct = self.msg_direct(
            (post_x, user_x),
            edge_index_dict[('post', 'rev_engages', 'user')]
        )
        
        # Social graph messages: users → users via follow relationships
        msg_social = self.msg_social(
            (user_x, user_x),
            edge_index_dict[('user', 'social', 'user')]
        )
        
        # Weighted combination of direct and social signals for user representations
        user_out = F.relu(self.w_direct * msg_direct + self.w_social * msg_social)
        
        # Update post representations using current user embeddings
        post_out = F.relu(
            self.post_update(
                (user_x, post_x),
                edge_index_dict[('user', 'engages', 'post')]
            )
        )
        
        return {'user': user_out, 'post': post_out}

# Instantiate model and load trained weights
model = WeightedRGCN(hidden_dim=64).to(device)
model.load_state_dict(torch.load("best_rgcn_model.pt", weights_only=True))
model.eval()  # Set to evaluation mode (disable dropout/batchnorm)

# Load precomputed post embeddings from trained model for efficient scoring
embed_data = torch.load("higgs_embeddings_trained.pt")
known_post_emb = embed_data['post_emb'].to(device)  # [num_posts, hidden_dim]

# Build tweet lookup dictionary mapping global post IDs to original text
activity_sub = data['activity_sub']
tweet_lookup = {}
for _, row in activity_sub.iterrows():
    global_id = data['post_to_idx'][row["post_id"]]  # Map local post ID to global node index
    tweet_lookup[global_id] = row["tweet"]

# Initialize Sentence-BERT encoder for text-based semantic matching (not used in final eval)
encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# ----------------------------
# Load FULL activity data (for future engagement check)
# ----------------------------
def load_truths_correct(tsv_file):
    """
    Load truths.tsv with explicit column mapping to handle inconsistent TSV formatting.
    Extracts key fields: ID, timestamp, author, likes, text, reply/quote flags, and IDs.
    
    Args:
        tsv_file (str): Path to truths.tsv
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with essential truth post metadata
    """
    records = []
    with open(tsv_file, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')  # Skip header row
        for line in f:
            parts = line.rstrip('\n').split('\t')
            # Require minimum 13 columns for safe indexing (schema: id, timestamp, ..., author, likes, ..., text, ...)
            if len(parts) >= 13:
                try:
                    id = parts[0]
                    timestamp = parts[1]
                    author = str(parts[5])          # Author handle
                    likes = int(parts[6]) if parts[6].isdigit() else 0  # Parse likes count
                    text = parts[9]                 # Tweet content
                    is_reply = parts[4]             # 't'/'f' flag for replies
                    quoted_truth_id = parts[12]     # ID of quoted post (if any)
                    truth_id = parts[10]            # Unique truth post ID
                    is_quote = parts[3]             # 't'/'f' flag for quotes
                    
                    external_id = str(parts[10])    # External identifier
                    
                    # Only include records with essential fields present
                    if author and external_id and text:
                        records.append({
                            'id': id,
                            'timestamp': timestamp,
                            'likes': likes,
                            'is_reply': is_reply,
                            'is_quote': is_quote,
                            'external_id': external_id,
                            'quoted_truth_id': quoted_truth_id,
                            'text': text,
                            'truth_id': truth_id,
                            'author': author
                        })
                except Exception:
                    # Skip malformed lines silently
                    continue
    return pd.DataFrame(records)

# Load and process full activity history for evaluation
truths = load_truths_correct("truth_social/truths.tsv")
truths_df = pd.DataFrame({
    'user_id': truths['author'],
    'created_at': truths['timestamp'],
    'interaction': 'POST',
    'likes': truths['likes'],
    'is_reply': truths['is_reply'],
    'is_quote': truths['is_quote'],
    'truth_id': truths['truth_id'],
    'quoted_truth_id': truths['quoted_truth_id'],
    'text': truths['text']
})
truths_df["created_at"] = pd.to_datetime(truths_df["created_at"], errors='coerce')

# Load explicit reply engagements
replies = pd.read_csv("truth_social/replies.tsv", sep="\t", dtype=str)
replies["timestamp"] = pd.to_datetime(replies["time_scraped"], errors='coerce')
replies = replies.rename(columns={
    "replying_user": "engager",   # User who made the reply
    "replied_user": "target_user" # User being replied to
})

# Match reply engagements to actual reply posts using hourly time buckets
reply_truths = truths_df[truths_df["is_reply"] == "t"].copy()
reply_truths["time_rounded"] = reply_truths["created_at"].dt.floor("H")
replies["time_rounded"] = replies["timestamp"].dt.floor("H")

reply_activity = replies.merge(
    reply_truths,
    left_on=["engager", "time_rounded"],
    right_on=["user_id", "time_rounded"],
    how="inner"
)
reply_activity = reply_activity.drop_duplicates(subset=["engager", "timestamp", "target_user"])
reply_activity = reply_activity[[
    "engager", "target_user", "timestamp", "text"
]].rename(columns={"text": "tweet"})
reply_activity["interaction"] = "RE"  # Label interaction type as "RE" (Reply)

# Load quote engagements
quotes_df = pd.read_csv("truth_social/quotes.tsv", sep="\t", dtype=str)
quote_activity = quotes_df.merge(
    truths_df,
    left_on="quoted_truth_external_id",
    right_on="truth_id",
    how="inner"
)
quote_activity = quote_activity[[
    "quoting_user", "quoted_user", "timestamp", "text"
]].rename(columns={
    "quoting_user": "engager",    # User who made the quote
    "quoted_user": "target_user", # User whose post was quoted
    "text": "tweet"
})
quote_activity["interaction"] = "QT"  # Label interaction type as "QT" (Quote)

# Combine all interaction types into unified activity timeline
full_activity = pd.concat([quote_activity, reply_activity], ignore_index=True)
full_activity["timestamp"] = pd.to_datetime(full_activity["timestamp"], errors="coerce")
full_activity = full_activity.dropna(subset=["timestamp"]).reset_index(drop=True)
full_activity = full_activity.sort_values("timestamp").reset_index(drop=True)

# ----------------------------
# Get test users (last 10% of activity)
# ----------------------------
# Split timeline: first 90% for training context, last 10% for evaluation
n = len(full_activity)
test_start = int(0.9 * n)
test_activity = full_activity.iloc[test_start:].copy()

# Identify users with future engagements in test period
test_engaged_users = set(test_activity["engager"])

# Load full user base from social graph for sampling context
all_users_df = pd.read_csv("truth_social/follows.tsv", sep="\t", usecols=["follower", "followed"])
all_users = set(all_users_df["follower"]) | set(all_users_df["followed"])

print(f"Total users in social graph: {len(all_users)}")
print(f"Users with future engagements: {len(test_engaged_users)}")

# Safety check: ensure sufficient test users available
if len(test_engaged_users) < 5:
    raise ValueError(f"Only {len(test_engaged_users)} users with future engagements — need at least 5")

# Randomly sample 29 test users with future engagements (seeded for reproducibility)
random.seed(42)
test_users_sample = random.sample(list(test_engaged_users), 29)
print(f"Selected 29 test users with future engagements: {test_users_sample}")

# ----------------------------
# Build candidate pool: ALL posts from training set
# ----------------------------
# Create candidate set from training period posts for recommendation
candidate_posts = []
for _, row in activity_sub.iterrows():
    candidate_posts.append({
        "id": str(row["post_id"]),   # Local post ID (matches training set indexing)
        "text": row["tweet"]         # Original tweet text
    })
print(f"Total candidate posts: {len(candidate_posts)}")

# ----------------------------
# Helper: Compute structural features for any user
# ----------------------------
def get_user_features(user_id, social_df, activity_df):
    """
    Compute structural user features for inductive inference:
    [log(followers+1), log(followees+1), log(engagement_count+1)]
    
    Args:
        user_id (str): User identifier
        social_df (pd.DataFrame): Full social graph edges
        activity_df (pd.DataFrame): Full activity history
    
    Returns:
        np.ndarray: 3-dimensional structural feature vector
    """
    # Compute in-degree (followers)
    in_deg = len(social_df[social_df['followee'] == user_id])
    # Compute out-degree (followees)
    out_deg = len(social_df[social_df['follower'] == user_id])
    # Compute engagement count (posts/replies/quotes made)
    engagement = len(activity_df[activity_df['engager'] == user_id])
    
    # Log-transform with +1 smoothing to handle zeros
    return np.array([
        np.log(in_deg + 1),
        np.log(out_deg + 1),
        np.log(engagement + 1)
    ], dtype=np.float32)

# ----------------------------
# Inductive recommendation function (works for ANY user)
# ----------------------------
def recommend_for_user_inductive(user_id, candidates, social_df, activity_df, K=10):
    """
    Generate post recommendations for ANY user (including unseen users) using inductive inference:
    1. Compute structural features from social graph and activity history
    2. Create minimal graph with user node and required edge types
    3. Pass through trained GNN to get user embedding
    4. Score against all known post embeddings using dot product
    5. Return top-K posts with highest similarity scores
    
    Args:
        user_id (str): Target user identifier
        candidates (list): List of candidate posts with IDs and text
        social_df (pd.DataFrame): Social graph edges for feature computation
        activity_df (pd.DataFrame): Activity history for engagement counting
        K (int): Number of recommendations to return
    
    Returns:
        tuple: (list of top-K posts, list of corresponding scores)
    """
    # Compute structural features for the user
    user_feat_np = get_user_features(user_id, social_df, activity_df)
    
    # Pad features to match post feature dimensionality (for compatibility with model)
    post_dim = x[num_users:].shape[1]  # Post feature dimension from training
    if len(user_feat_np) < post_dim:
        padding = np.zeros(post_dim - len(user_feat_np))
        user_feat_np = np.concatenate([user_feat_np, padding])
    
    # Convert to tensor and move to device
    user_feat = torch.tensor(user_feat_np, dtype=torch.float).unsqueeze(0).to(device)
    
    # Build minimal heterogeneous graph containing only the target user
    mini_graph = HeteroData()
    mini_graph['user'].x = user_feat  # Single user node with structural features
    mini_graph['post'].x = torch.empty(0, post_dim).to(device)  # No post nodes needed (using precomputed embeddings)
    
    # Add all edge types expected by model (empty tensors since no connections exist in minimal graph)
    empty_edge = torch.empty(2, 0, dtype=torch.long).to(device)
    mini_graph['user', 'social', 'user'].edge_index = empty_edge
    mini_graph['user', 'engages', 'post'].edge_index = empty_edge
    mini_graph['post', 'authored_by', 'user'].edge_index = empty_edge
    mini_graph['post', 'rev_engages', 'user'].edge_index = empty_edge
    
    # Generate user embedding using trained GNN (inductive inference)
    with torch.no_grad():
        out = model({'user': mini_graph['user'].x, 'post': mini_graph['post'].x}, mini_graph.edge_index_dict)
        user_emb = out['user']  # [1, 64] embedding vector
    
    # Score user against all known posts using dot product (cosine similarity since embeddings are normalized)
    scores = torch.mm(user_emb, known_post_emb.T).squeeze(0)
    topk_scores, topk_indices = torch.topk(scores, min(K, len(scores)))
    
    # Map top-K indices back to candidate posts
    top_posts = []
    top_scores = []
    for idx, score in zip(topk_indices.cpu().numpy(), topk_scores.cpu().numpy()):
        # Find candidate with matching local post ID (idx corresponds to training set post index)
        for cand in candidates:
            if cand["id"] == str(idx):
                top_posts.append(cand)
                top_scores.append(score.item())
                break
    
    return top_posts, top_scores

# ----------------------------
# Evaluation
# ----------------------------
print("\n" + "="*60)
print("INDUCTIVE EVALUATION: 29 USERS WITH FUTURE ENGAGEMENTS")
print("="*60)

# Load social graph for structural feature computation during evaluation
social_df = pd.read_csv("truth_social/follows.tsv", sep="\t", dtype=str)
social_df.rename(columns={"followed": "followee"}, inplace=True)

# Evaluate each test user
for i, user_id in enumerate(test_users_sample, 1):
    print(f"\n{i}. USER {user_id}")
    
    # Get ground truth future engagements for this user (from test period)
    true_engagements = test_activity[test_activity["engager"] == user_id]
    true_texts = set(true_engagements["tweet"].tolist())
    print(f"   True future engagements: {len(true_texts)}")
    
    # Generate recommendations using inductive inference
    rec_posts, rec_scores = recommend_for_user_inductive(
        user_id, candidate_posts, social_df, full_activity, K=10
    )
    print(f"   Recommendations: {len(rec_posts)}")
    
    # Compute Recall@10: proportion of future engagements captured in top-10 recommendations
    rec_texts = set([p["text"] for p in rec_posts])
    hits = len(true_texts & rec_texts)  # Intersection of recommended and actual future posts
    recall = hits / len(true_texts) if true_texts else 0
    print(f"   Hits: {hits}, Recall@10: {recall:.4f}")
    
    # Display examples for qualitative assessment
    print("   ✅ True future (first 2):")
    for text in list(true_texts)[:2]:
        print(f"     - {text[:80]}...")
    
    print("   🎯 Top recommendations (first 3):")
    for post, score in zip(rec_posts[:3], rec_scores[:3]):
        print(f"     - {post['text'][:80]}... (score: {score:.4f})")

