'''
import pandas as pd
import numpy as np
import torch

social = pd.read_csv("higgs_social_network.edgelist", sep=" ", header=None, names=["follower", "followee"])
activity = pd.read_csv("higgs_activity_time.txt", sep=" ", header=None, names=["engager", "target_user", "timestamp", "interaction"])

user_counts = pd.concat([activity["engager"], activity["target_user"]]).value_counts()
top_users = user_counts.head(5000).index

activity_sub = activity[
    activity["engager"].isin(top_users) & activity["target_user"].isin(top_users)
].copy().head(10000).reset_index(drop=True)

# CRITICAL: post_id = 0,1,2,...,9999 (local)
activity_sub["post_id"] = activity_sub.index

social_sub = social[
    social["follower"].isin(top_users) & social["followee"].isin(top_users)
].copy()

users = sorted(set(activity_sub["engager"]) | set(activity_sub["target_user"]) | set(social_sub["follower"]) | set(social_sub["followee"]))
U = len(users)
P = len(activity_sub)

user_to_idx = {u: i for i, u in enumerate(users)}
# post_to_idx: local (0-9999) → global (U to U+9999)
post_to_idx = {i: U + i for i in range(P)}

# Build edges (global IDs)
social_mapped = social_sub.copy()
social_mapped["follower"] = social_mapped["follower"].map(user_to_idx)
social_mapped["followee"] = social_mapped["followee"].map(user_to_idx)
social_mapped = social_mapped.dropna().astype(int)

edge_index_social = torch.tensor(social_mapped[["follower", "followee"]].values.T, dtype=torch.long)

engagement = activity_sub[["engager", "post_id"]].copy()
engagement["engager"] = engagement["engager"].map(user_to_idx)
engagement["post_id"] = engagement["post_id"].map(post_to_idx)  # local→global
engagement = engagement.dropna().astype(int)
edge_index_engage = torch.tensor(engagement[["engager", "post_id"]].values.T, dtype=torch.long)

authorship = activity_sub[["post_id", "target_user"]].copy()
authorship["post_id"] = authorship["post_id"].map(post_to_idx)
authorship["target_user"] = authorship["target_user"].map(user_to_idx)
authorship = authorship.dropna().astype(int)
edge_index_author = torch.tensor(authorship[["post_id", "target_user"]].values.T, dtype=torch.long)

# Features
in_social = np.zeros(U); out_social = np.zeros(U); engagement_count = np.zeros(U)
for _, row in social_mapped.iterrows():
    f, t = int(row["follower"]), int(row["followee"])
    out_social[f] += 1; in_social[t] += 1

eng_counts = activity_sub["engager"].map(user_to_idx).value_counts()
for uid, cnt in eng_counts.items():
    engagement_count[int(uid)] = cnt

user_features = torch.tensor(np.stack([
    np.log(in_social + 1), np.log(out_social + 1), np.log(engagement_count + 1)
], axis=1), dtype=torch.float)

post_features = torch.zeros(P, 3)
interaction_map = {"RT": 0, "RE": 1, "MT": 2}
for i, inter in enumerate(activity_sub["interaction"]):
    post_features[i, interaction_map[inter]] = 1.0

x = torch.cat([user_features, post_features], dim=0)

torch.save({
    'user_to_idx': user_to_idx,
    'post_to_idx': post_to_idx,
    'x': x,
    'edge_index_social': edge_index_social,
    'activity_sub': activity_sub,
    'num_users': U,
    'num_posts': P,
}, "higgs_processed.pt")
'''


'''
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import json

# ----------------------------
# 1. Load synthetic activity data (must include 'tweet' column)
# ----------------------------
activity = pd.read_csv("synthetic_higgs_full_all_users.csv")
required_cols = ["engager", "target_user", "timestamp", "interaction", "tweet"]
activity = activity[required_cols].copy()

# ----------------------------
# 2. Load real social network
# ----------------------------
social = pd.read_csv("higgs_social_network.edgelist", sep=" ", header=None, names=["follower", "followee"])

# ----------------------------
# 3. Subset to top 5K users (for training feasibility)
# ----------------------------
user_counts = pd.concat([activity["engager"], activity["target_user"]]).value_counts()
top_users = user_counts.head(5000).index

activity_sub = activity[
    activity["engager"].isin(top_users) & 
    activity["target_user"].isin(top_users)
].copy().head(10000).reset_index(drop=True)
activity_sub["post_id"] = activity_sub.index

# ----------------------------
# 4. Load user_to_topic mapping (from clustering on FULL graph)
# ----------------------------
with open("user_to_topic_full.json", "r") as f:
    user_to_topic_full = json.load(f)
# Convert keys to int
user_to_topic_full = {int(k): int(v) for k, v in user_to_topic_full.items()}

# ----------------------------
# 5. Filter SOCIAL EDGES to be within-topic (using full mapping)
# ----------------------------
social_sub = social[
    social["follower"].isin(top_users) & 
    social["followee"].isin(top_users)
]

# Keep only edges where follower and followee share the same topic
def same_topic(row):
    follower = row["follower"]
    followee = row["followee"]
    # Both users must be in the full mapping (they should be)
    if follower not in user_to_topic_full or followee not in user_to_topic_full:
        return False
    return user_to_topic_full[follower] == user_to_topic_full[followee]

#social_sub = social_sub[social_sub.apply(same_topic, axis=1)].copy()
#print(f"Filtered social edges: kept {len(social_sub)} edges")

# ----------------------------
# 6. Build node mappings
# ----------------------------
users = sorted(
    set(activity_sub["engager"]) | 
    set(activity_sub["target_user"]) | 
    set(social_sub["follower"]) | 
    set(social_sub["followee"])
)
U = len(users)
P = len(activity_sub)

user_to_idx = {u: i for i, u in enumerate(users)}
post_to_idx = {local_id: U + local_id for local_id in range(P)}

# ----------------------------
# 7. Build edge indices
# ----------------------------
# Social edges
social_mapped = social_sub.copy()
social_mapped["follower"] = social_mapped["follower"].map(user_to_idx)
social_mapped["followee"] = social_mapped["followee"].map(user_to_idx)
social_mapped = social_mapped.dropna().astype(int)
edge_index_social = torch.tensor(social_mapped[["follower", "followee"]].values.T, dtype=torch.long)

# Engagement edges (user → post)
engagement = activity_sub[["engager", "post_id"]].copy()
engagement["engager"] = engagement["engager"].map(user_to_idx)
engagement["post_id"] = engagement["post_id"].map(post_to_idx)
engagement = engagement.dropna().astype(int)
edge_index_engage = torch.tensor(engagement[["engager", "post_id"]].values.T, dtype=torch.long)

# Authorship edges (post → user)
authorship = activity_sub[["post_id", "target_user"]].copy()
authorship["post_id"] = authorship["post_id"].map(post_to_idx)
authorship["target_user"] = authorship["target_user"].map(user_to_idx)
authorship = authorship.dropna().astype(int)
edge_index_author = torch.tensor(authorship[["post_id", "target_user"]].values.T, dtype=torch.long)

# Build: for each post, who follows its author?
author_of_post = {}  # post_local_id → author_user_id
for i in range(graph['post', 'authored_by', 'user'].edge_index.shape[1]):
    post_local = graph['post', 'authored_by', 'user'].edge_index[0, i].item()
    author = graph['post', 'authored_by', 'user'].edge_index[1, i].item()
    author_of_post[post_local] = author

# Build reverse social: user → set of users they follow
follows = {}
for i in range(graph['user', 'social', 'user'].edge_index.shape[1]):
    follower = graph['user', 'social', 'user'].edge_index[0, i].item()
    followee = graph['user', 'social', 'user'].edge_index[1, i].item()
    if follower not in follows:
        follows[follower] = set()
    follows[follower].add(followee)

# Build followed_by edges: (post, user) if user follows author of post
src_posts, dst_users = [], []
for post_local, author in author_of_post.items():
    for user in range(num_users):
        if user in follows and author in follows[user]:
            src_posts.append(post_local)
            dst_users.append(user)

if src_posts:
    graph['post', 'followed_by', 'user'].edge_index = torch.tensor([src_posts, dst_users], dtype=torch.long)
else:
    # Fallback: empty edge
    graph['post', 'followed_by', 'user'].edge_index = torch.empty((2, 0), dtype=torch.long)

# ----------------------------
# 8. Build node features
# ----------------------------
in_social = np.zeros(U)
out_social = np.zeros(U)
engagement_count = np.zeros(U)

for _, row in social_mapped.iterrows():
    f, t = int(row["follower"]), int(row["followee"])
    out_social[f] += 1
    in_social[t] += 1

eng_counts = activity_sub["engager"].map(user_to_idx).value_counts()
for uid, cnt in eng_counts.items():
    engagement_count[int(uid)] = cnt

user_features = torch.tensor(np.stack([
    np.log(in_social + 1),
    np.log(out_social + 1),
    np.log(engagement_count + 1)
], axis=1), dtype=torch.float)

# Post features
print("Encoding tweet semantics...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
tweets = activity_sub["tweet"].tolist()
tweet_embeddings = encoder.encode(tweets, show_progress_bar=True)
tweet_embeddings = torch.tensor(tweet_embeddings, dtype=torch.float)

interaction_map = {"RT": 0, "RE": 1, "MT": 2}
interaction_feats = torch.zeros(P, 3)
for i, inter in enumerate(activity_sub["interaction"]):
    if inter in interaction_map:
        interaction_feats[i, interaction_map[inter]] = 1.0

post_features = torch.cat([interaction_feats, tweet_embeddings], dim=1)

# Pad user features
post_feat_dim = post_features.size(1)
user_feat_dim = user_features.size(1)
if user_feat_dim < post_feat_dim:
    padding = torch.zeros(U, post_feat_dim - user_feat_dim)
    user_features_padded = torch.cat([user_features, padding], dim=1)
else:
    user_features_padded = user_features

x = torch.cat([user_features_padded, post_features], dim=0)

# ----------------------------
# 9. Save processed graph
# ----------------------------
torch.save({
    'user_to_idx': user_to_idx,
    'post_to_idx': post_to_idx,
    'x': x,
    'edge_index_social': edge_index_social,
    'activity_sub': activity_sub,
    'num_users': U,
    'num_posts': P,
}, "synthetic_processed_with_semantics.pt")

print(f"✅ Processed synthetic data with semantics:")
print(f"   Users: {U}")
print(f"   Posts: {P}")
print(f"   Social edges: {edge_index_social.shape[1]}")
print(f"   Post feature dim: {post_features.shape[1]}")
print(f"   Saved to: synthetic_processed_with_semantics.pt")
'''


import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import json

# ----------------------------
# 1. Load synthetic activity data
# ----------------------------
replies = pd.read_csv(
    "truth_social/replies.tsv",
    sep="\t",
    usecols=["replying_user", "replied_user", "time_scraped"]
)

replies = replies.rename(columns={
    "replying_user": "engager",
    "replied_user": "target_user",
    "time_scraped": "timestamp"
})

replies["interaction"] = "RE"
replies["tweet"] = ""   # no reliable text join


quotes = pd.read_csv("truth_social/quotes.tsv", sep="\t")
# Map to your schema
quotes = quotes.rename(columns={
    "quoting_user": "engager",
    "quoted_user": "target_user"
})
quotes["interaction"] = "QT"
quotes["timestamp"] = quotes["timestamp"]  # quotes has actual timestamp
activity_raw = pd.concat([replies, quotes], ignore_index=True)

# ----------------------------
# 1. Load Truth Social activity (FIXED)
# ----------------------------

# Load truths (authoritative source of text + authorship)
def load_truths_correct(tsv_file):
    records = []
    with open(tsv_file, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 13:
                try:
                    id = parts[0]
                    author = str(parts[5])      # ← CONVERT TO STRING
                    text = parts[9]
                    external_id = str(parts[10])  # ← CONVERT TO STRING
                    if author and external_id and text:
                        records.append({
                            'id': id,
                            'external_id': external_id,
                            'text': text,
                            'author': author
                        })
                except IndexError:
                    continue
    return pd.DataFrame(records)

truths = load_truths_correct("truth_social/truths.tsv")

# Load quotes
quotes = pd.read_csv(
    "truth_social/quotes.tsv",
    sep="\t",
    usecols=[
        "quoting_user",
        "quoted_user",
        "quoted_truth_external_id",
        "timestamp"
    ]
)

# Join quotes → truths USING external_id (CRITICAL FIX)
quotes['quoted_truth_external_id'] = quotes['quoted_truth_external_id'].astype(str)
#truths['external_id'] = truths['external_id'].astype(str)
quotes['quoting_user'] = quotes['quoting_user'].astype(str)
quotes['quoted_user'] = quotes['quoted_user'].astype(str)


activity = quotes.merge(
    truths,
    left_on="quoted_truth_external_id",
    right_on="external_id",
    how="inner"
)

# Map to synthetic schema
activity = activity.rename(columns={
    "quoting_user": "engager",
    "author": "target_user",
    "text": "tweet"
})

activity["interaction"] = "QT"

# Final selection
activity = activity[
    ["engager", "target_user", "timestamp", "interaction", "tweet"]
].dropna().reset_index(drop=True)

# ----------------------------
# 2. Load real social network
# ----------------------------
# Replace your current social loading
social = pd.read_csv("truth_social/follows.tsv", sep="\t", usecols=["follower", "followed"])
social['follower'] = social['follower'].astype(str)
social['followed'] = social['followed'].astype(str)
social.rename(columns={"followed": "followee"}, inplace=True)

activity = pd.concat([activity, replies], ignore_index=True)
#activity = activity[["engager", "target_user", "timestamp", "interaction", "tweet"]].dropna().reset_index(drop=True)

# ----------------------------
# 3. Subset to top 5K users
# ----------------------------
user_counts = pd.concat([activity["engager"], activity["target_user"]]).value_counts()
top_users = user_counts.head(5000).index

activity_sub = activity[
    activity["engager"].isin(top_users) & 
    activity["target_user"].isin(top_users)
].copy().head(10000).reset_index(drop=True)
activity_sub["post_id"] = activity_sub.index
#social['follower'] = social['follower'].astype(str)
#social['followee'] = social['followee'].astype(str)
activity_sub['engager'] = activity_sub['engager'].astype(str)
activity_sub['target_user'] = activity_sub['target_user'].astype(str)
# ----------------------------
# 4. Build node mappings
# ----------------------------
users = sorted(
    set(activity_sub["engager"]) | 
    set(activity_sub["target_user"]) | 
    set(social["follower"]) | 
    set(social["followee"])
)
U = len(users)
P = len(activity_sub)

user_to_idx = {u: i for i, u in enumerate(users)}
post_to_idx = {local_id: U + local_id for local_id in range(P)}

# ----------------------------
# 5. Build base edge indices
# ----------------------------
# Social edges (user → user)
social_sub = social[
    social["follower"].isin(top_users) & 
    social["followee"].isin(top_users)
].copy()
social_mapped = social_sub.copy()
social_mapped["follower"] = social_mapped["follower"].map(user_to_idx)
social_mapped["followee"] = social_mapped["followee"].map(user_to_idx)
social_mapped = social_mapped.dropna().astype(int)
edge_index_social = torch.tensor(social_mapped[["follower", "followee"]].values.T, dtype=torch.long)

# Engagement edges (user → post)
engagement = activity_sub[["engager", "post_id"]].copy()
engagement["engager"] = engagement["engager"].map(user_to_idx)
engagement["post_id"] = engagement["post_id"].map(post_to_idx)
engagement = engagement.dropna().astype(int)
edge_index_engage = torch.tensor(engagement[["engager", "post_id"]].values.T, dtype=torch.long)

# Authorship edges (post → user)
authorship = activity_sub[["post_id", "target_user"]].copy()
authorship["post_id"] = authorship["post_id"].map(post_to_idx)
authorship["target_user"] = authorship["target_user"].map(user_to_idx)
authorship = authorship.dropna().astype(int)
edge_index_author = torch.tensor(authorship[["post_id", "target_user"]].values.T, dtype=torch.long)

# ----------------------------
# 6. BUILD FOLLOWED_BY EDGE: (post → user) if user follows the author of the post
# ----------------------------
print("Building 'followed_by' edge: user follows author of post...")

# Step 1: Map post (global ID) → author (user local ID)
post_global_to_author = {}
for _, row in authorship.iterrows():
    post_global = int(row["post_id"])
    author_local = int(row["target_user"])
    post_global_to_author[post_global] = author_local

# Step 2: Build set of who each user follows (local IDs)
follows_set = {}
for _, row in social_mapped.iterrows():
    follower = int(row["follower"])
    followee = int(row["followee"])
    if follower not in follows_set:
        follows_set[follower] = set()
    follows_set[follower].add(followee)

# Step 3: For each post, find users who follow its author
src_posts = []  # post global IDs
dst_users = []  # user local IDs

for post_global, author_local in post_global_to_author.items():
    for user_local in range(U):  # all users
        if user_local in follows_set and author_local in follows_set[user_local]:
            src_posts.append(post_global)
            dst_users.append(user_local)

# Convert to edge index (post → user)
if src_posts:
    edge_index_followed_by = torch.tensor([src_posts, dst_users], dtype=torch.long)
else:
    edge_index_followed_by = torch.empty((2, 0), dtype=torch.long)

print(f"Built 'followed_by' edge with {edge_index_followed_by.shape[1]} connections")

# ----------------------------
# 7. Build node features
# ----------------------------
in_social = np.zeros(U)
out_social = np.zeros(U)
engagement_count = np.zeros(U)

for _, row in social_mapped.iterrows():
    f, t = int(row["follower"]), int(row["followee"])
    out_social[f] += 1
    in_social[t] += 1

eng_counts = activity_sub["engager"].map(user_to_idx).value_counts()
for uid, cnt in eng_counts.items():
    engagement_count[int(uid)] = cnt

user_features = torch.tensor(np.stack([
    np.log(in_social + 1),
    np.log(out_social + 1),
    np.log(engagement_count + 1)
], axis=1), dtype=torch.float)

# Post features
print("Encoding tweet semantics...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
tweets = activity_sub["tweet"].tolist()
tweet_embeddings = encoder.encode(tweets, show_progress_bar=True)
tweet_embeddings = torch.tensor(tweet_embeddings, dtype=torch.float)

interaction_map = {"QT": 0, "RE": 1}  # Add QT, remove unused RT/MT
interaction_feats = torch.zeros(P, 3)
for i, inter in enumerate(activity_sub["interaction"]):
    if inter in interaction_map:
        interaction_feats[i, interaction_map[inter]] = 1.0

post_features = torch.cat([interaction_feats, tweet_embeddings], dim=1)

# Pad user features
post_feat_dim = post_features.size(1)
user_feat_dim = user_features.size(1)
if user_feat_dim < post_feat_dim:
    padding = torch.zeros(U, post_feat_dim - user_feat_dim)
    user_features_padded = torch.cat([user_features, padding], dim=1)
else:
    user_features_padded = user_features

x = torch.cat([user_features_padded, post_features], dim=0)

# ----------------------------
# 8. Save processed graph
# ----------------------------
torch.save({
    'user_to_idx': user_to_idx,
    'post_to_idx': post_to_idx,
    'x': x,
    'edge_index_social': edge_index_social,
    'edge_index_engage': edge_index_engage,
    'edge_index_author': edge_index_author,
    'edge_index_followed_by': edge_index_followed_by,  # ← NEW!
    'activity_sub': activity_sub,
    'num_users': U,
    'num_posts': P,
}, "synthetic_processed_with_semantics.pt")

print(f"✅ Processed synthetic data with semantics:")
print(f"   Users: {U}")
print(f"   Posts: {P}")
print(f"   Social edges: {edge_index_social.shape[1]}")
print(f"   Followed-by edges: {edge_index_followed_by.shape[1]}")
print(f"   Saved to: synthetic_processed_with_semantics.pt")