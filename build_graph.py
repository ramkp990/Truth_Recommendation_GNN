import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import re
import unicodedata
from urllib.parse import urlparse
import csv
import sys
import torch.nn.functional as F


# ----------------------------
# Text Cleaning & URL Enrichment
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
    text = '. '.join(unique_sentences) + ('.' if unique_sentences else '')
    
    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle very short/empty results after cleaning
    if not text or len(text) < 3:
        return "[EMPTY]"
    
    # Truncate extremely long texts to 512 characters
    return text[:512]

# ----------------------------
# Load Data
# ----------------------------
print("Loading quotes...")
quotes_df = pd.read_csv("truth_social/quotes.tsv", sep="\t", dtype=str)

# Load truths with correct schema
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

truths = load_truths_correct("truth_social/truths.tsv")
print(f"Loaded {len(truths)} truths")

# Create original posts DataFrame with standardized column names
truths_df = pd.DataFrame({
    'user_id': truths['author'],
    'created_at': truths['timestamp'],
    'likes': truths['likes'],
    'is_reply': truths['is_reply'],
    'is_quote': truths['is_quote'],
    'truth_id': truths['truth_id'],
    'quoted_truth_id': truths['quoted_truth_id'],
    'text': truths['text']
})
# Convert timestamp strings to datetime objects for time-based operations
truths_df["created_at"] = pd.to_datetime(truths_df["created_at"], errors='coerce')

# ----------------------------
# Build Reply Activity
# ----------------------------
# Filter truths that are replies (is_reply='t')
reply_truths = truths_df[truths_df["is_reply"] == "t"].copy()
print(f"Reply posts in truths.tsv: {len(reply_truths)}")

# Load explicit reply engagements from replies.tsv
replies = pd.read_csv("truth_social/replies.tsv", sep="\t", dtype=str)
replies["timestamp"] = pd.to_datetime(replies["time_scraped"], errors='coerce')  # Standardize timestamp column
replies = replies.rename(columns={
    "replying_user": "engager",   # User who made the reply
    "replied_user": "target_user" # User being replied to
})
print(f"Replies in replies.tsv: {len(replies)}")

# Fuzzy temporal join: match reply engagements to actual reply posts using hourly time buckets
# (Accounts for slight timestamp mismatches between datasets)
reply_truths["time_rounded"] = reply_truths["created_at"].dt.floor("H")  # Round to nearest hour
replies["time_rounded"] = replies["timestamp"].dt.floor("H")

# Inner join on engager + rounded timestamp to link engagements with actual posts
reply_activity = replies.merge(
    reply_truths,
    left_on=["engager", "time_rounded"],
    right_on=["user_id", "time_rounded"],
    how="inner"
)
# Remove duplicate engagements (same user replying multiple times in same hour)
reply_activity = reply_activity.drop_duplicates(subset=["engager", "timestamp", "truth_id", "target_user"])
reply_activity = reply_activity[[
    "engager", "target_user", "timestamp", "truth_id", "text"
]].rename(columns={"text": "tweet"})  # Standardize column name for tweet content
reply_activity["interaction"] = "RE"  # Label interaction type as "RE" (Reply)
print(f"Matched reply engagements: {len(reply_activity)}")

# ----------------------------
# Build Quote Activity (FIXED: only one merge)
# ----------------------------
print("Building quote activity...")
# Direct join: match quotes to original posts using quoted_truth_external_id → truth_id
# (More reliable than temporal joins since quote explicitly references target post ID)
quote_activity = quotes_df.merge(
    truths_df,
    left_on="quoted_truth_external_id", 
    right_on="truth_id",
    how="inner"
)
# Standardize columns and label interaction type
quote_activity = quote_activity[[
    "quoting_user", "quoted_user", "timestamp", "truth_id", "text"
]].rename(columns={
    "quoting_user": "engager",    # User who made the quote
    "quoted_user": "target_user", # User whose post was quoted
    "text": "tweet"
})
quote_activity["interaction"] = "QT"  # Label interaction type as "QT" (Quote)
print(f"Quote activity shape: {len(quote_activity)}")

# ----------------------------
# Build POST Activity (Original Posts)
# ----------------------------
# Filter truths that are neither quotes nor replies → original content posts
original_posts = truths_df[
    (truths_df["is_quote"] == "f") & 
    (truths_df["is_reply"] == "f")
].copy()

# Create POST activity records: user posts their own original content
post_activity = original_posts.copy()
post_activity = post_activity.rename(columns={
    "user_id": "engager",      # Poster is both engager and target
    "created_at": "timestamp",
    "text": "tweet"
})
post_activity["target_user"] = post_activity["engager"]  # Self-referential for original posts
post_activity["interaction"] = "POST"  # Label interaction type as "POST"
post_activity = post_activity[[
    "engager", "target_user", "timestamp", "truth_id", "tweet"
]]
print(f"Original post activity: {len(post_activity)}")

# ----------------------------
# Combine All Activities
# ----------------------------
# Merge all three interaction types into single activity log
activity = pd.concat([quote_activity, reply_activity, post_activity], ignore_index=True)
print(f"Combined activity (QT + RE + POST): {len(activity)}")

# Standardize and clean timestamps across all activities
activity["timestamp"] = pd.to_datetime(activity["timestamp"], errors="coerce")
activity = activity.dropna(subset=["timestamp"]).reset_index(drop=True)  # Drop rows with invalid timestamps
print(f"After timestamp cleaning: {len(activity)}")

# ----------------------------
# Load Social Graph
# ----------------------------
print("Loading social graph...")
# Load follower-followee relationships (directed edges: follower → followee)
social = pd.read_csv(
    "truth_social/follows.tsv",
    sep="\t",
    usecols=["follower", "followed"],  # Only load essential columns
    dtype=str
)
social.rename(columns={"followed": "followee"}, inplace=True)  # Standardize column name

# ----------------------------
# Select Active Users
# ----------------------------
print("Selecting active users...")

# Count engagement metrics per user (as engager)
quoters = activity[activity["interaction"] == "QT"].groupby("engager").size()
repliers = activity[activity["interaction"] == "RE"].groupby("engager").size()
posters = activity[activity["interaction"] == "POST"].groupby("engager").size()

# Count engagement metrics per user (as target)
quoted_users = activity[activity["interaction"] == "QT"].groupby("target_user").size()
replied_users = activity[activity["interaction"] == "RE"].groupby("target_user").size()
posted_users = activity[activity["interaction"] == "POST"].groupby("target_user").size()

# Compute social graph degrees
out_deg = social.groupby("follower").size()   # Out-degree: number of followees
in_deg = social.groupby("followee").size()    # In-degree: number of followers

# Aggregate all observed users across activities and social graph
all_users = set(quoters.index) | set(repliers.index) | set(posters.index) | \
           set(quoted_users.index) | set(replied_users.index) | set(posted_users.index) | \
           set(out_deg.index) | set(in_deg.index)

# Build user feature matrix with engagement/social metrics
user_features = pd.DataFrame(index=sorted(all_users))
user_features["n_quotes"] = quoters.reindex(user_features.index, fill_value=0)
user_features["n_replies"] = repliers.reindex(user_features.index, fill_value=0)
user_features["n_posts"] = posters.reindex(user_features.index, fill_value=0)
user_features["n_quoted"] = quoted_users.reindex(user_features.index, fill_value=0)
user_features["n_replied"] = replied_users.reindex(user_features.index, fill_value=0)
user_features["n_posted"] = posted_users.reindex(user_features.index, fill_value=0)
user_features["n_followees"] = out_deg.reindex(user_features.index, fill_value=0)
user_features["n_followers"] = in_deg.reindex(user_features.index, fill_value=0)
user_features.reset_index(inplace=True)
user_features.rename(columns={"index": "engager"}, inplace=True)

# Active user selection criteria:
# - Made at least 1 quote OR reply OR
# - Was quoted/replied to/posted at least once AND has ≥1 follower
active_mask = (
    (user_features["n_quotes"] >= 1) | 
    (user_features["n_replies"] >= 1) |
    (
        (user_features["n_quoted"] >= 1) | 
        (user_features["n_replied"] >= 1) |
        (user_features["n_posted"] >= 1)
    ) &
    (user_features["n_followers"] >= 1)
)

# Select top 5,000 most active users (prioritizing quoters for research focus)
candidate_users = user_features[active_mask].sort_values("n_quotes", ascending=False)
active_users = set(candidate_users.head(5000)["engager"])
print(f"Selected {len(active_users)} active users.")

# ----------------------------
# Filter Activity & Cap Size
# ----------------------------
# Filter activity log to only include interactions between active users
activity_sub = activity[
    activity["engager"].isin(active_users) &
    activity["target_user"].isin(active_users)
].reset_index(drop=True)

# Prioritize high-value interactions (QT/RE) over POSTs for dataset balance
qt_re_activity = activity_sub[activity_sub["interaction"].isin(["QT", "RE"])]
post_activity_sub = activity_sub[activity_sub["interaction"] == "POST"]

# Cap QT/RE interactions to 15,000 samples
if len(qt_re_activity) > 15000:
    qt_re_activity = qt_re_activity.sample(n=15000, random_state=42)
remaining_budget = 20000 - len(qt_re_activity)

# Fill remaining budget with POST interactions (max 5,000)
if len(post_activity_sub) > remaining_budget:
    post_activity_sub = post_activity_sub.sample(n=remaining_budget, random_state=42)

# Combine capped interaction types into final activity subset
activity_sub = pd.concat([qt_re_activity, post_activity_sub], ignore_index=True).reset_index(drop=True)
activity_sub["post_id"] = activity_sub.index  # Assign unique post ID for graph construction
print(f"Final activity shape: {len(activity_sub)}")

# ----------------------------
# Build Graph
# ----------------------------
# Create unified node index: users first, then posts
users = sorted(set(activity_sub["engager"]) | set(activity_sub["target_user"]))
U = len(users)  # Number of user nodes
P = len(activity_sub)  # Number of post nodes

user_to_idx = {u: i for i, u in enumerate(users)}  # Map user ID → node index (0 to U-1)
post_to_idx = {i: U + i for i in range(P)}  # Map post ID → node index (U to U+P-1)

# Social edges: follower → followee (user-to-user)
social_sub = social[
    social["follower"].isin(users) & 
    social["followee"].isin(users)
]
social_mapped = social_sub.copy()
social_mapped["follower"] = social_mapped["follower"].map(user_to_idx)
social_mapped["followee"] = social_mapped["followee"].map(user_to_idx)
social_mapped = social_mapped.dropna().astype(int)  # Drop unmapped users
edge_index_social = torch.tensor(social_mapped[["follower", "followee"]].values.T, dtype=torch.long)

# Engagement edges: user → post (who interacted with which post)
engagement = activity_sub[["engager", "post_id"]].copy()
engagement["engager"] = engagement["engager"].map(user_to_idx)
engagement["post_id"] = engagement["post_id"].map(post_to_idx)
engagement = engagement.dropna().astype(int)
edge_index_engage = torch.tensor(engagement[["engager", "post_id"]].values.T, dtype=torch.long)

# Authorship edges: post → user (which user authored which post)
# Note: For POST interactions, target_user = engager (self-authorship)
authorship = activity_sub[["post_id", "target_user"]].copy()
authorship["post_id"] = authorship["post_id"].map(post_to_idx)
authorship["target_user"] = authorship["target_user"].map(user_to_idx)
authorship = authorship.dropna().astype(int)
edge_index_author = torch.tensor(authorship[["post_id", "target_user"]].values.T, dtype=torch.long)

# ----------------------------
# Node Features
# ----------------------------
# User features: log-transformed in-degree, out-degree, and engagement count
in_social = np.zeros(U)
out_social = np.zeros(U)
engagement_count = np.zeros(U)

# Compute social graph degrees
for _, row in social_mapped.iterrows():
    f, t = int(row["follower"]), int(row["followee"])
    out_social[f] += 1  # Out-degree increment
    in_social[t] += 1   # In-degree increment

# Compute engagement counts per user
eng_counts = activity_sub["engager"].map(user_to_idx).value_counts()
for uid, cnt in eng_counts.items():
    engagement_count[int(uid)] = cnt

# Stack features and apply log-transform for normalization
user_features_tensor = torch.tensor(np.stack([
    np.log(in_social + 1),        # +1 to avoid log(0)
    np.log(out_social + 1),
    np.log(engagement_count + 1)
], axis=1), dtype=torch.float)

# Post features: semantic embeddings + interaction type indicators
print("Encoding enriched tweet semantics...")
activity_sub["tweet_clean"] = activity_sub["tweet"].apply(clean_tweet_text)  # Preprocess text
encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight sentence transformer
tweets_clean = activity_sub["tweet_clean"].tolist()
tweet_embeddings = encoder.encode(tweets_clean, show_progress_bar=True)  # Generate 384-dim embeddings
tweet_embeddings = torch.tensor(tweet_embeddings, dtype=torch.float)

# One-hot encode interaction types (QT, RE, POST) as 3-dimensional indicator vector
interaction_feats = torch.zeros(P, 3)
qt_mask = activity_sub["interaction"] == "QT"
re_mask = activity_sub["interaction"] == "RE"
post_mask = activity_sub["interaction"] == "POST"
interaction_feats[qt_mask, 0] = 1.0  # First dim = QT
interaction_feats[re_mask, 1] = 1.0  # Second dim = RE
interaction_feats[post_mask, 2] = 1.0  # Third dim = POST

# Concatenate interaction type indicators with semantic embeddings
post_features = torch.cat([interaction_feats, tweet_embeddings], dim=1)

# Project both user and post features to 64-dimensional space with random projections
# (For demonstration purposes - in practice, use learned projections or dimensionality reduction)
proj_user = torch.randn(3, 64)   # 3 user features → 64 dims
proj_post = torch.randn(387, 64) # 3 (interaction) + 384 (embedding) = 387 → 64 dims

user_features_64 = F.normalize(torch.mm(user_features_tensor, proj_user), dim=1)
post_features_64 = F.normalize(torch.mm(post_features, proj_post), dim=1)

# Alternative approach: pad user features to match post feature dimensionality (387)
# (Commented out since we're using projection approach above)
# post_feat_dim = post_features.size(1)
# user_feat_dim = user_features_tensor.size(1)
# if user_feat_dim < post_feat_dim:
#     padding = torch.zeros(U, post_feat_dim - user_feat_dim)
#     user_features_padded = torch.cat([user_features_tensor, padding], dim=1)
# else:
#     user_features_padded = user_features_tensor

# Final node feature matrix: [user_features; post_features] stacked vertically
# Using projected 64-dim features for both node types
x = torch.cat([user_features_64, post_features_64], dim=0)

# ----------------------------
# Save
# ----------------------------
# Save complete graph structure and metadata for model training
torch.save({
    'user_to_idx': user_to_idx,           # User ID → node index mapping
    'post_to_idx': post_to_idx,           # Post ID → node index mapping
    'x': x,                               # Node feature matrix (U+P nodes × 64 features)
    'edge_index_social': edge_index_social,  # Social graph edges (follower → followee)
    'edge_index_engage': edge_index_engage,  # Engagement edges (user → post)
    'edge_index_author': edge_index_author,  # Authorship edges (post → user)
    'activity_sub': activity_sub,         # Original activity dataframe for reference
    'num_users': U,                       # Number of user nodes
    'num_posts': P,                       # Number of post nodes
}, "synthetic_processed_with_semantics.pt")

print(f"\n✅ Saved graph:")
print(f"   Users: {U}")
print(f"   Posts: {P}")
print(f"   Social edges: {edge_index_social.shape[1]}")
print(f"   Engagement edges: {edge_index_engage.shape[1]}")
print(f"   Authorship edges: {edge_index_author.shape[1]}")

# Save model configuration metadata for reproducibility
import json
with open("model_config.json", "w") as f:
    json.dump({
        "text_encoder": "all-MiniLM-L6-v2",
        "max_post_length": 512,
        "feature_dim": post_features.shape[1],  # Original feature dimension before projection
        "interaction_types": ["QT", "RE", "POST"]
    }, f)

print("Saved model configuration.")
