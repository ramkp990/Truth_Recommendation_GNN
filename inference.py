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
    text = re.sub(r'https?\s*:\s*//', 'https://', text, flags=re.IGNORECASE)
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    if not urls:
        return text
    main_url = urls[0]
    parsed = urlparse(main_url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    keywords = [word for word in re.split(r'[-_/]', path) if word.isalpha() and len(word) > 2]
    keyword_str = " ".join(keywords[:5])
    if keyword_str:
        return f"[LINK] {domain} {keyword_str}"
    else:
        return f"[LINK] {domain}"

def clean_tweet_text(text):
    if not isinstance(text, str) or not text.strip():
        return "[EMPTY]"
    url_count = len(re.findall(r'https?://', text))
    word_count = len(re.findall(r'\b\w+\b', text))
    if url_count >= 1 and word_count <= 5:
        return extract_url_features(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'<emoji:\s*\w+\s*>', ' [EMOJI] ', text)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    unique_sentences = []
    for s in sentences:
        if not unique_sentences or s != unique_sentences[-1]:
            unique_sentences.append(s)
    text = '. '.join(unique_sentences) + ('.' if unique_sentences else '')
    text = re.sub(r'\s+', ' ', text).strip()
    if not text or len(text) < 3:
        return "[EMPTY]"
    return text[:512]

# ----------------------------
# Load trained model and data
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load graph data (for post features and social graph structure)
data = torch.load("synthetic_processed_with_semantics.pt", weights_only=False)
x = data['x']
num_users = data['num_users']
num_posts = data['num_posts']

# Load trained model
from torch_geometric.nn import SAGEConv

class WeightedRGCN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.msg_direct = SAGEConv((-1, -1), hidden_dim)
        self.msg_social = SAGEConv((-1, -1), hidden_dim)
        self.post_update = SAGEConv((-1, -1), hidden_dim)
        self.w_direct = 1.0
        self.w_social = 0.75

    def forward(self, x_dict, edge_index_dict):
        user_x, post_x = x_dict['user'], x_dict['post']
        msg_direct = self.msg_direct(
            (post_x, user_x),
            edge_index_dict[('post', 'rev_engages', 'user')]
        )
        msg_social = self.msg_social(
            (user_x, user_x),
            edge_index_dict[('user', 'social', 'user')]
        )
        user_out = F.relu(self.w_direct * msg_direct + self.w_social * msg_social)
        post_out = F.relu(
            self.post_update(
                (user_x, post_x),
                edge_index_dict[('user', 'engages', 'post')]
            )
        )
        return {'user': user_out, 'post': post_out}

model = WeightedRGCN(hidden_dim=64).to(device)
model.load_state_dict(torch.load("best_rgcn_model.pt", weights_only=True))
model.eval()

# Load post embeddings (for candidate scoring)
embed_data = torch.load("higgs_embeddings_trained.pt")
known_post_emb = embed_data['post_emb'].to(device)

# Load activity for post text lookup
activity_sub = data['activity_sub']
tweet_lookup = {}
for _, row in activity_sub.iterrows():
    global_id = data['post_to_idx'][row["post_id"]]
    tweet_lookup[global_id] = row["tweet"]

# Initialize SBERT encoder
encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# ----------------------------
# Load FULL activity data (for future engagement check)
# ----------------------------
def load_truths_correct(tsv_file):
    records = []
    with open(tsv_file, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 13:
                try:
                    id = parts[0]
                    timestamp = parts[1]
                    author = str(parts[5])
                    likes = int(parts[6]) if parts[6].isdigit() else 0
                    text = parts[9]
                    is_reply = parts[4]
                    quoted_truth_id = parts[12]
                    truth_id = parts[10]
                    is_quote = parts[3]
                    external_id = str(parts[10])
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
                    continue
    return pd.DataFrame(records)

# Load truths
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

# Load replies
replies = pd.read_csv("truth_social/replies.tsv", sep="\t", dtype=str)
replies["timestamp"] = pd.to_datetime(replies["time_scraped"], errors='coerce')
replies = replies.rename(columns={
    "replying_user": "engager",
    "replied_user": "target_user"
})

# Match replies to posts
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
reply_activity["interaction"] = "RE"

# Load quotes
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
    "quoting_user": "engager",
    "quoted_user": "target_user",
    "text": "tweet"
})
quote_activity["interaction"] = "QT"

# Combine all activity
full_activity = pd.concat([quote_activity, reply_activity], ignore_index=True)
full_activity["timestamp"] = pd.to_datetime(full_activity["timestamp"], errors="coerce")
full_activity = full_activity.dropna(subset=["timestamp"]).reset_index(drop=True)
full_activity = full_activity.sort_values("timestamp").reset_index(drop=True)

# ----------------------------
# Get test users (last 10% of activity)
# ----------------------------
n = len(full_activity)
test_start = int(0.9 * n)
test_activity = full_activity.iloc[test_start:].copy()

# Users with future engagements (CORRECTED: only those with actual future engagements)
test_engaged_users = set(test_activity["engager"])

# Load full user base from follows
all_users_df = pd.read_csv("truth_social/follows.tsv", sep="\t", usecols=["follower", "followed"])
all_users = set(all_users_df["follower"]) | set(all_users_df["followed"])

print(f"Total users in social graph: {len(all_users)}")
print(f"Users with future engagements: {len(test_engaged_users)}")

# Ensure we have at least 5 engaged users
if len(test_engaged_users) < 5:
    raise ValueError(f"Only {len(test_engaged_users)} users with future engagements — need at least 5")

# Randomly sample 5 engaged users (from full population, but guaranteed to have future engagements)
random.seed(42)
test_users_sample = random.sample(list(test_engaged_users), 29)
print(f"Selected 5 test users with future engagements: {test_users_sample}")

# ----------------------------
# Build candidate pool: ALL posts from training set
# ----------------------------
candidate_posts = []
for _, row in activity_sub.iterrows():
    candidate_posts.append({
        "id": str(row["post_id"]),
        "text": row["tweet"]
    })
print(f"Total candidate posts: {len(candidate_posts)}")

# ----------------------------
# Helper: Compute structural features for any user
# ----------------------------
def get_user_features(user_id, social_df, activity_df):
    """Compute [log(followers+1), log(followees+1), log(engagement+1)]"""
    in_deg = len(social_df[social_df['followee'] == user_id])
    out_deg = len(social_df[social_df['follower'] == user_id])
    engagement = len(activity_df[activity_df['engager'] == user_id])
    return np.array([
        np.log(in_deg + 1),
        np.log(out_deg + 1),
        np.log(engagement + 1)
    ], dtype=np.float32)

# ----------------------------
# Inductive recommendation function (works for ANY user)
# ----------------------------
def recommend_for_user_inductive(user_id, candidates, social_df, activity_df, K=10):
    """Recommend posts for ANY user using inductive inference"""
    # Compute structural features
    user_feat_np = get_user_features(user_id, social_df, activity_df)
    
    # Pad to match post feature dimension
    post_dim = x[num_users:].shape[1]
    if len(user_feat_np) < post_dim:
        padding = np.zeros(post_dim - len(user_feat_np))
        user_feat_np = np.concatenate([user_feat_np, padding])
    
    # Create temporary user node
    user_feat = torch.tensor(user_feat_np, dtype=torch.float).unsqueeze(0).to(device)
    
    # Build minimal graph with ALL required edge types
    mini_graph = HeteroData()
    mini_graph['user'].x = user_feat
    mini_graph['post'].x = torch.empty(0, post_dim).to(device)
    
    # Add ALL edge types expected by the model (even if empty)
    empty_edge = torch.empty(2, 0, dtype=torch.long).to(device)
    mini_graph['user', 'social', 'user'].edge_index = empty_edge
    mini_graph['user', 'engages', 'post'].edge_index = empty_edge
    mini_graph['post', 'authored_by', 'user'].edge_index = empty_edge
    mini_graph['post', 'rev_engages', 'user'].edge_index = empty_edge
    
    # Get embedding via trained GNN
    with torch.no_grad():
        out = model({'user': mini_graph['user'].x, 'post': mini_graph['post'].x}, mini_graph.edge_index_dict)
        user_emb = out['user']  # [1, 64]
    
    # Score against all known posts
    scores = torch.mm(user_emb, known_post_emb.T).squeeze(0)
    topk_scores, topk_indices = torch.topk(scores, min(K, len(scores)))
    
    # Map back to candidate posts
    top_posts = []
    top_scores = []
    for idx, score in zip(topk_indices.cpu().numpy(), topk_scores.cpu().numpy()):
        # Find candidate with matching local post ID
        for cand in candidates:
            if cand["id"] == str(idx):  # post_id = local index
                top_posts.append(cand)
                top_scores.append(score.item())
                break
    
    return top_posts, top_scores

# ----------------------------
# Evaluation
# ----------------------------
print("\n" + "="*60)
print("INDUCTIVE EVALUATION: 5 USERS WITH FUTURE ENGAGEMENTS")
print("="*60)

# Load social graph for feature computation
social_df = pd.read_csv("truth_social/follows.tsv", sep="\t", dtype=str)
social_df.rename(columns={"followed": "followee"}, inplace=True)

for i, user_id in enumerate(test_users_sample, 1):
    print(f"\n{i}. USER {user_id}")
    
    # Get true future engagements
    true_engagements = test_activity[test_activity["engager"] == user_id]
    true_texts = set(true_engagements["tweet"].tolist())
    print(f"   True future engagements: {len(true_texts)}")
    
    # Get recommendations (inductive for ANY user)
    rec_posts, rec_scores = recommend_for_user_inductive(
        user_id, candidate_posts, social_df, full_activity, K=10
    )
    print(f"   Recommendations: {len(rec_posts)}")
    
    # Compute recall
    rec_texts = set([p["text"] for p in rec_posts])
    hits = len(true_texts & rec_texts)
    recall = hits / len(true_texts) if true_texts else 0
    print(f"   Hits: {hits}, Recall@10: {recall:.4f}")
    
    # Show examples
    print("   ✅ True future (first 2):")
    for text in list(true_texts)[:2]:
        print(f"     - {text[:80]}...")
    
    print("   🎯 Top recommendations (first 3):")
    for post, score in zip(rec_posts[:3], rec_scores[:3]):
        print(f"     - {post['text'][:80]}... (score: {score:.4f})")