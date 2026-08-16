import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load clusters
clusters_df = pd.read_csv("/kaggle/input/datasets/akshit1616/new-clusters4-5-wordbank/clusters_5.csv")
# Ensure the column is named 'cluster_id' for internal consistency
if 'cluster' in clusters_df.columns:
    clusters_df.rename(columns={'cluster': 'cluster_id'}, inplace=True)

# Load word bank phrases
with open("/kaggle/input/datasets/akshit1616/new-clusters4-5-wordbank/wordbank.txt", "r") as f:
    phrases = [line.strip() for line in f if line.strip()]

print("Loading Waterbirds dataset...")
# Load the combined dataset so image indexing matches your previous steps
dataset = load_dataset("grodino/waterbirds", split="train+validation+test")

# ==========================================
# 2. Setup ID Mapping (Reserved 0, 1, 2, 3)
# ==========================================
# Extract valid clusters (ignore noise which is usually -1)
unique_valid_clusters = sorted([cid for cid in clusters_df['cluster_id'].unique() if cid != -1])

# Map old Role B cluster IDs to new Role C slice IDs starting at 4
id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_valid_clusters, start=4)}
print(f"Remapping cluster IDs to avoid collision with base groups: {id_mapping}")

# ==========================================
# 3. Initialize CLIP
# ==========================================
print("Initializing CLIP model...")
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()

# ==========================================
# 4. Generate & Mean-Center Image Embeddings
# ==========================================
print(f"Generating embeddings for {len(clusters_df)} images...")
image_ids = clusters_df['image_id'].tolist()
image_embeddings = []

with torch.no_grad():
    for img_id in tqdm(image_ids, desc="Images"):
        img = dataset[int(img_id)]['image'].convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        
        # Correct function: only runs the vision half of the model
        image_embeds = model.get_image_features(**inputs)
        
        # Ensure it remains a 2D tensor [batch_size, hidden_dim]
        if not isinstance(image_embeds, torch.Tensor):
            # In case an older Transformers version returns an object
            image_embeds = getattr(image_embeds, 'image_embeds', getattr(image_embeds, 'pooler_output', image_embeds[0]))
        
        if image_embeds.dim() == 1:
            image_embeds = image_embeds.unsqueeze(0)
            
        image_embeddings.append(image_embeds.cpu())

# Stack into [N, 512] and L2 normalize
image_embeddings = torch.cat(image_embeddings, dim=0)
image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

# Mean center the embeddings
global_mean = image_embeddings.mean(dim=0, keepdim=True)
centered_image_embeddings = image_embeddings - global_mean
centered_image_embeddings = F.normalize(centered_image_embeddings, p=2, dim=-1)

# ==========================================
# 5. Generate Text Embeddings for Word Bank
# ==========================================
print(f"Generating text embeddings for {len(phrases)} phrases...")
text_embeddings = []
batch_size = 128

with torch.no_grad():
    for i in tqdm(range(0, len(phrases), batch_size), desc="Text"):
        batch_phrases = phrases[i:i+batch_size]
        inputs = processor(text=batch_phrases, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Correct function: only runs the text half of the model
        text_embeds = model.get_text_features(**inputs)
        
        if not isinstance(text_embeds, torch.Tensor):
            text_embeds = getattr(text_embeds, 'text_embeds', getattr(text_embeds, 'pooler_output', text_embeds[0]))
            
        if text_embeds.dim() == 1:
            text_embeds = text_embeds.unsqueeze(0)
            
        text_embeddings.append(text_embeds.cpu())

# Stack into [V, 512] and L2 normalize
text_embeddings = torch.cat(text_embeddings, dim=0)
text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
# ==========================================
# 6. Baseline & Contrastive Scoring
# ==========================================
print("Scoring clusters and generating names...")
slice_names_data = []

for old_id in unique_valid_clusters:
    new_id = id_mapping[old_id]
    
    # Identify indices for images inside and outside the current cluster
    in_cluster_mask = (clusters_df['cluster_id'] == old_id).values
    out_cluster_mask = (clusters_df['cluster_id'] != old_id).values
    
    # 1. Cluster centroid (average embedding of images IN the cluster)
    cluster_emb = centered_image_embeddings[in_cluster_mask].mean(dim=0, keepdim=True)
    cluster_emb = F.normalize(cluster_emb, p=2, dim=-1)
    
    # 2. Out-of-cluster centroid (average embedding of all OTHER images)
    other_emb = centered_image_embeddings[out_cluster_mask].mean(dim=0, keepdim=True)
    other_emb = F.normalize(other_emb, p=2, dim=-1)
    
    # Baseline Metric: Nearest-caption similarity (Cluster -> Text)
    cluster_sims = (cluster_emb @ text_embeddings.T).squeeze()
    
    # Contrastive Metric: (Cluster -> Text) MINUS (Other -> Text)
    other_sims = (other_emb @ text_embeddings.T).squeeze()
    contrastive_scores = cluster_sims - other_sims
    
    # Extract the Top 5 Phrases based on the contrastive score
    top_indices = contrastive_scores.argsort(descending=True)[:5]
    top_scores = contrastive_scores[top_indices]
    top_5_phrases = [phrases[i] for i in top_indices]
    
    # Append with the NEW mapped ID
    slice_names_data.append({
        'cluster_id': new_id,
        'top_phrase': top_5_phrases[0],
        'score': round(top_scores[0].item(), 4),
        'top_5_phrases': ", ".join(top_5_phrases)
    })

# ==========================================
# 7. Export Hand-off Artifacts
# ==========================================
# Artifact 1: slice_names.csv 
slice_names_df = pd.DataFrame(slice_names_data)
slice_names_df.to_csv("slice_names.csv", index=False)
print("\nSaved slice_names.csv! Preview:")
print(slice_names_df.head())

# Artifact 2: slice_membership.csv 
# Filter out the noise class (-1)
slice_membership_df = clusters_df[clusters_df['cluster_id'] != -1][['image_id', 'cluster_id']].copy()
# Apply the mapping to shift the IDs
slice_membership_df['cluster_id'] = slice_membership_df['cluster_id'].map(id_mapping)
# Rename to match handoff schema
slice_membership_df.rename(columns={'cluster_id': 'slice_id'}, inplace=True)

slice_membership_df.to_csv("slice_membership.csv", index=False)
print("\nSaved slice_membership.csv! Preview:")
print(slice_membership_df.head())


