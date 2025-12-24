import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.image.tokenizer import MedicalImageTokenizer, get_model, DEVICE

# --- CONFIGURATION ---
DATASET_ROOT = Path("dataset")
TRAIN_CSV = DATASET_ROOT / "train.csv"
VALID_CSV = DATASET_ROOT / "valid.csv"

# Number of samples to use for the experiment
DB_SIZE = 500       # Number of images in the database (from train)
QUERY_SIZE = 50     # Number of query images (from valid)
TOP_K = 5

# Labels to consider for medical similarity
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

def get_image_path(csv_path):
    """Converts CSV path to local path."""
    # CSV path: CheXpert-v1.0-small/train/patient...
    # Local path: dataset/train/patient...
    parts = csv_path.split("/")
    if len(parts) > 1:
        # Remove the first component (CheXpert-v1.0-small)
        relative_path = "/".join(parts[1:])
        return DATASET_ROOT / relative_path
    return DATASET_ROOT / csv_path

def load_data(csv_file, sample_size=None):
    df = pd.read_csv(csv_file)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    valid_data = []
    for idx, row in df.iterrows():
        img_path = get_image_path(row['Path'])
        if img_path.exists():
            # Extract labels (1.0 means positive)
            labels = []
            for lbl in LABELS:
                if row[lbl] == 1.0:
                    labels.append(lbl)
            
            valid_data.append({
                "id": idx,
                "path": str(img_path),
                "labels": labels
            })
    return valid_data

def get_embedding(image_path):
    """Extracts raw embedding using the same model as Tokenizer."""
    try:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path)
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        model = get_model()
        with torch.no_grad():
            embedding = model(img_tensor).flatten().cpu().numpy()
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def calculate_metrics(proposed_indices, baseline_indices, k):
    """
    Calculates Precision@k, Recall@k, nDCG@k assuming Baseline is Ground Truth.
    """
    proposed_set = set(proposed_indices[:k])
    baseline_set = set(baseline_indices[:k])
    
    # Precision@k: Fraction of proposed items that are in baseline
    intersection = len(proposed_set.intersection(baseline_set))
    precision = intersection / k
    
    # Recall@k: Fraction of baseline items found in proposed
    recall = intersection / k # Since both sets have size k, Precision == Recall here
    
    # nDCG@k
    # Relevance: 1 if item is in baseline top-k, 0 otherwise
    dcg = 0.0
    idcg = 0.0
    
    for i in range(k):
        # IDCG (Ideal): All top-k items are relevant
        idcg += 1.0 / np.log2(i + 2)
        
        # DCG (Actual)
        item_id = proposed_indices[i]
        if item_id in baseline_set:
            dcg += 1.0 / np.log2(i + 2)
            
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return precision, recall, ndcg

def calculate_label_utility(retrieved_indices, query_labels, db_data):
    """
    Calculates the proportion of retrieved items that share at least one label with the query.
    """
    if not query_labels:
        return 0.0 # Cannot match if query has no labels
        
    match_count = 0
    for idx in retrieved_indices:
        retrieved_item = db_data[idx]
        # Check intersection of labels
        common = set(query_labels).intersection(set(retrieved_item['labels']))
        if common:
            match_count += 1
            
    return match_count / len(retrieved_indices)

def main():
    print("--- Experiment RQ1: On-device PSI vs Baseline ---")
    
    # 1. Load Data
    print(f"Loading Database (size={DB_SIZE})...")
    db_data = load_data(TRAIN_CSV, DB_SIZE)
    print(f"Loaded {len(db_data)} valid DB images.")
    
    print(f"Loading Queries (size={QUERY_SIZE})...")
    query_data = load_data(VALID_CSV, QUERY_SIZE)
    print(f"Loaded {len(query_data)} valid Query images.")
    
    tokenizer = MedicalImageTokenizer()
    
    # 2. Precompute Embeddings and Tokens for DB
    print("Precomputing DB features...")
    db_embeddings = []
    db_tokens = []
    valid_db_indices = [] # To map back to db_data
    
    for i, item in enumerate(tqdm(db_data)):
        emb = get_embedding(item['path'])
        toks = tokenizer.process(item['path'])
        
        if emb is not None and toks:
            db_embeddings.append(emb)
            db_tokens.append(set(toks)) # Use set for faster intersection
            valid_db_indices.append(i)
            
    db_embeddings = np.array(db_embeddings)
    # Update db_data to only include valid ones
    db_data = [db_data[i] for i in valid_db_indices]
    
    print(f"Final DB size: {len(db_data)}")
    
    # 3. Run Experiment
    metrics = {
        "precision": [],
        "recall": [],
        "ndcg": [],
        "baseline_utility": [],
        "proposed_utility": []
    }
    
    print("Running Queries...")
    for q_item in tqdm(query_data):
        q_emb = get_embedding(q_item['path'])
        q_toks = tokenizer.process(q_item['path'])
        
        if q_emb is None or not q_toks:
            continue
            
        # --- BASELINE (Cosine Similarity) ---
        # Reshape q_emb to (1, D)
        sims = cosine_similarity(q_emb.reshape(1, -1), db_embeddings)[0]
        # Get indices of top-k (descending order)
        baseline_top_k_indices = np.argsort(sims)[::-1][:TOP_K]
        
        # --- PROPOSED (Token Intersection) ---
        q_toks_set = set(q_toks)
        scores = []
        for db_tok_set in db_tokens:
            # Intersection count
            score = len(q_toks_set.intersection(db_tok_set))
            scores.append(score)
        
        scores = np.array(scores)
        # Get indices of top-k (descending order)
        # Note: If scores are equal, argsort might not be stable, but acceptable for exp
        proposed_top_k_indices = np.argsort(scores)[::-1][:TOP_K]
        
        # --- EVALUATION ---
        p, r, n = calculate_metrics(proposed_top_k_indices, baseline_top_k_indices, TOP_K)
        
        # Label Utility
        base_util = calculate_label_utility(baseline_top_k_indices, q_item['labels'], db_data)
        prop_util = calculate_label_utility(proposed_top_k_indices, q_item['labels'], db_data)
        
        metrics["precision"].append(p)
        metrics["recall"].append(r)
        metrics["ndcg"].append(n)
        metrics["baseline_utility"].append(base_util)
        metrics["proposed_utility"].append(prop_util)
        
    # 4. Report
    print("\n--- Results (Average @ K={}) ---".format(TOP_K))
    print(f"Precision: {np.mean(metrics['precision']):.4f}")
    print(f"Recall:    {np.mean(metrics['recall']):.4f}")
    print(f"nDCG:      {np.mean(metrics['ndcg']):.4f}")
    print("-" * 30)
    print(f"Baseline Label Utility: {np.mean(metrics['baseline_utility']):.4f}")
    print(f"Proposed Label Utility: {np.mean(metrics['proposed_utility']):.4f}")
    
    # Interpretation
    print("\n--- Interpretation ---")
    print("Precision/Recall/nDCG measure how well the Proposed method approximates the Baseline (Plaintext).")
    print("Label Utility measures how medically relevant the retrieved results are (do they share the same diagnosis?).")

if __name__ == "__main__":
    main()
