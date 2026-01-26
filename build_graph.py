import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import re
import unicodedata
from urllib.parse import urlparse

# ----------------------------
# Text Cleaning & URL Enrichment
# ----------------------------
def extract_url_features(text):
    # Fix malformed URLs like "https  ://"
    text = re.sub(r'https?\s*:\s*//', 'https://', text, flags=re.IGNORECASE)
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    
    if not urls:
        return text
    
    main_url = urls[0]
    parsed = urlparse(main_url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    
    # Extract keywords from path
    keywords = [word for word in re.split(r'[-_/]', path) if word.isalpha() and len(word) > 2]
    keyword_str = " ".join(keywords[:5])
    
    if keyword_str:
        return f"[LINK] {domain} {keyword_str}"
    else:
        return f"[LINK] {domain}"

def clean_tweet_text(text):
    if not isinstance(text, str) or not text.strip():
        return "[EMPTY]"
    
    # Count URLs and words
    url_count = len(re.findall(r'https?://', text))
    word_count = len(re.findall(r'\b\w+\b', text))
    
    # Enrich URL-only posts
    if url_count >= 1 and word_count <= 5:
        return extract_url_features(text)
    
    # Normalize and clean regular text
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'<emoji:\s*\w+\s*>', ' [EMOJI] ', text)
    
    # Remove duplicate sentences
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
# Load Data
# ----------------------------
print("Loading quotes...")
quotes_df = pd.read_csv("truth_social/quotes.tsv", sep="\t", dtype=str)



# Load truths with correct schema
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

truths = load_truths_correct("truth_social/truths.tsv")

# Create original posts
truths = pd.DataFrame({
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
print(truths.head(20))
truths["created_at"] = pd.to_datetime(truths["created_at"], errors='coerce')




# Keep only reply posts
reply_truths = truths[truths["is_reply"] == "t"].copy()
print(f"Reply posts in truths.tsv: {len(reply_truths)}")

replies = pd.read_csv("truth_social/replies.tsv", sep="\t", dtype=str)
replies["timestamp"] = pd.to_datetime(replies["time_scraped"], errors='coerce')
replies = replies.rename(columns={
    "replying_user": "engager",
    "replied_user": "target_user"
})
print(f"Replies in replies.tsv: {len(replies)}")

# ----------------------------
# 3. Fuzzy join: match reply posts to reply actions
# ----------------------------
# We'll merge on:
#   - engager (from replies) == user_id (from truths)
#   - timestamps within 1 hour (to handle scraping lag)

# Add rounded time for robust matching
reply_truths["time_rounded"] = reply_truths["created_at"].dt.floor("H")
replies["time_rounded"] = replies["timestamp"].dt.floor("H")

# Merge
reply_activity = replies.merge(
    reply_truths,
    left_on=["engager", "time_rounded"],
    right_on=["user_id", "time_rounded"],
    how="inner"
)

# If too many matches, refine with exact time or drop duplicates
reply_activity = reply_activity.drop_duplicates(subset=["engager", "timestamp", "target_user"])

print(f"Matched reply engagements: {len(reply_activity)}")

# ----------------------------
# 4. Build clean activity DataFrame
# ----------------------------
reply_activity = reply_activity[[
    "engager",
    "target_user",
    "timestamp",
    "text"
]].rename(columns={"text": "tweet"})
reply_activity["interaction"] = "RE"

print("Sample reply activity:")
print(reply_activity.head())





# ----------------------------
# Build Quote-Based Activity
# ----------------------------
print("Building quote activity...")
quote_activity = quotes_df.merge(
    truths,
    left_on="quoted_truth_external_id",
    right_on="truth_id",
    how="inner"
)
print(quote_activity.tail(10))
quote_activity = quote_activity[[
    "quoting_user",
    "quoted_user",          # ← this is the target user
    "timestamp",
    "text"
]].copy()

# Rename to your schema
quote_activity = quote_activity.rename(columns={
    "quoting_user": "engager",
    "quoted_user": "target_user",   # ← use quoted_user, not author
    "text": "tweet"
})
quote_activity["interaction"] = "QT"
print(quote_activity.head(10))
quote_activity["interaction"] = "QT"
print(f"Quote activity shape: {quote_activity.shape}")





activity = pd.concat([quote_activity, reply_activity], ignore_index=True)
print(f"Combined activity (QT + RE): {len(activity)}")

'''
# ----------------------------
# Deduplication
# ----------------------------
print("Deduplicating by normalized text...")
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().strip().split())

activity["text_norm"] = activity["tweet"].apply(normalize_text)
activity = activity.drop_duplicates(subset=["text_norm"], keep="first").reset_index(drop=True)
activity.drop(columns=["text_norm"], inplace=True)
'''
print(f"Combined activity (QT + RE): {len(activity)}")

activity = pd.concat([quote_activity, reply_activity], ignore_index=True)
print(f"Combined activity (QT + RE): {len(activity)}")

# ✅ CRITICAL: Normalize timestamp to datetime
activity["timestamp"] = pd.to_datetime(activity["timestamp"], errors="coerce")
activity = activity.dropna(subset=["timestamp"]).reset_index(drop=True)
print(f"After timestamp cleaning: {len(activity)}")


# ----------------------------
# Load Social Graph
# ----------------------------
print("Loading social graph...")
social = pd.read_csv(
    "truth_social/follows.tsv",
    sep="\t",
    usecols=["follower", "followed"],
    dtype=str
)
social.rename(columns={"followed": "followee"}, inplace=True)

# ----------------------------
# Select Active Users
# ----------------------------
print("Selecting active users...")

# Engagement stats (only quoters)
quoters = activity.groupby("engager").size()
quoted_users = activity.groupby("target_user").size()

# Social degrees
out_deg = social.groupby("follower").size()
in_deg = social.groupby("followee").size()

# Combine user sets
all_users = set(quoters.index) | set(quoted_users.index) | set(out_deg.index) | set(in_deg.index)
user_features = pd.DataFrame(index=sorted(all_users))
user_features["n_quotes"] = quoters.reindex(user_features.index, fill_value=0)
user_features["n_quoted"] = quoted_users.reindex(user_features.index, fill_value=0)
user_features["n_followees"] = out_deg.reindex(user_features.index, fill_value=0)
user_features["n_followers"] = in_deg.reindex(user_features.index, fill_value=0)
user_features.reset_index(inplace=True)
user_features.rename(columns={"index": "engager"}, inplace=True)

# Active user criteria
active_mask = (
    (user_features["n_quotes"] >= 1) |  # Quoted ≥5 times
    (
        (user_features["n_quoted"] >= 1) &  # Was quoted ≥10 times
        (user_features["n_followers"] >= 1)  # And has followers
    )
)

# Cap to top 5K most active
candidate_users = user_features[active_mask].sort_values("n_quotes", ascending=False)
active_users = set(candidate_users.head(5000)["engager"])
print(f"Selected {len(active_users)} active users.")

# ----------------------------
# Filter Activity & Cap Size
# ----------------------------
activity_sub = activity[
    activity["engager"].isin(active_users) &
    activity["target_user"].isin(active_users)
].reset_index(drop=True)

# Cap to 20K for trainability
if len(activity_sub) > 20000:
    activity_sub = activity_sub.sample(n=20000, random_state=42).reset_index(drop=True)

activity_sub["post_id"] = activity_sub.index
print(f"Final activity shape: {activity_sub.shape}")

# ----------------------------
# Build Graph
# ----------------------------
# Only include users in activity
users = sorted(set(activity_sub["engager"]) | set(activity_sub["target_user"]))
U = len(users)
P = len(activity_sub)

user_to_idx = {u: i for i, u in enumerate(users)}
post_to_idx = {i: U + i for i in range(P)}

# Social edges
social_sub = social[
    social["follower"].isin(users) & 
    social["followee"].isin(users)
]
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
# Node Features
# ----------------------------
# User features
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

user_features_tensor = torch.tensor(np.stack([
    np.log(in_social + 1),
    np.log(out_social + 1),
    np.log(engagement_count + 1)
], axis=1), dtype=torch.float)

# Post features
print("Encoding enriched tweet semantics...")
activity_sub["tweet_clean"] = activity_sub["tweet"].apply(clean_tweet_text)
encoder = SentenceTransformer('all-MiniLM-L6-v2')
tweets_clean = activity_sub["tweet_clean"].tolist()
tweet_embeddings = encoder.encode(tweets_clean, show_progress_bar=True)
tweet_embeddings = torch.tensor(tweet_embeddings, dtype=torch.float)

# Interaction type (only QT)
interaction_feats = torch.zeros(P, 2)
interaction_feats[:, 0] = 1.0  # All are QT

post_features = torch.cat([interaction_feats, tweet_embeddings], dim=1)

# Pad user features
post_feat_dim = post_features.size(1)
user_feat_dim = user_features_tensor.size(1)
if user_feat_dim < post_feat_dim:
    padding = torch.zeros(U, post_feat_dim - user_feat_dim)
    user_features_padded = torch.cat([user_features_tensor, padding], dim=1)
else:
    user_features_padded = user_features_tensor

x = torch.cat([user_features_padded, post_features], dim=0)

# ----------------------------
# Save
# ----------------------------
torch.save({
    'user_to_idx': user_to_idx,
    'post_to_idx': post_to_idx,
    'x': x,
    'edge_index_social': edge_index_social,
    'edge_index_engage': edge_index_engage,
    'edge_index_author': edge_index_author,
    'activity_sub': activity_sub,
    'num_users': U,
    'num_posts': P,
}, "synthetic_processed_with_semantics.pt")

print(f"\n✅ Saved graph:")
print(f"   Users: {U}")
print(f"   Posts: {P}")
print(f"   Social edges: {edge_index_social.shape[1]}")