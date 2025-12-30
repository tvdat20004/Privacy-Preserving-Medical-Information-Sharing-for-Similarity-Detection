import os
import sys
import time
import json
import requests
import csv
import shutil
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from client.image.tokenizer import MedicalImageTokenizer
from apsi import LabeledClient
from common.config import APSI_PARAMS

# --- CONFIGURATION ---
SERVER_DATA_DIR = _project_root / "server"
DATASET_CSV = _project_root / "dataset" / "valid.csv"
LOG_FILE = _project_root / "experiment/result/experiment_rq3_accuracy.csv"
OUTPUT_IMAGE = _project_root / "experiment/result/figure_4.3_accuracy.png"
TOP_K_VALUES = [5, 10, 20]

def load_ground_truth(data_folder_path):
    """
    Loads the server metadata from the specific data folder.
    """
    metadata_path = data_folder_path / "metadata.json"
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    gt = {}
    for rid, info in data.items():
        labels = set(info.get('diagnosis', []))
        gt[rid] = labels
    return gt

def setup_server_db(source_folder_name):
    """
    Copies the content of the selected DB folder to the active server/data folder
    and triggers a reload.
    """
    source_path = SERVER_DATA_DIR / source_folder_name
    target_path = SERVER_DATA_DIR / "data"
    
    print(f"Swapping DB: {source_folder_name} -> data")
    
    # 1. Clear target
    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.mkdir()
    
    # 2. Copy files
    # We only need metadata.json, tokens.json, medical.db
    for filename in ["metadata.json", "tokens.json", "medical.db"]:
        src_file = source_path / filename
        if src_file.exists():
            shutil.copy2(src_file, target_path / filename)
        else:
            print(f"Warning: {filename} missing in {source_folder_name}")
            
    # 3. Trigger Reload
    try:
        # Ensure we are not using a proxy that might cache or fail
        session = requests.Session()
        session.trust_env = False
        resp = session.post("http://localhost:5000/admin/reload_db", timeout=30)
        resp.raise_for_status()
        print("Server reloaded successfully.")
        return True
    except Exception as e:
        print(f"Failed to reload server: {e}")
        return False

def run_experiment_for_db(db_folder_name, precomputed_queries):
    """
    Runs experiment for a single DB size sequentially against the running server.
    """
    db_size = int(db_folder_name.split('_')[1])
    server_url = "http://localhost:5000"
    
    # 1. Setup DB
    if not setup_server_db(db_folder_name):
        print(f"Skipping {db_folder_name} due to setup failure.")
        return []
    
    # 2. Load Ground Truth (from source folder, not active folder, to be safe)
    db_folder_path = SERVER_DATA_DIR / db_folder_name
    gt_db = load_ground_truth(db_folder_path)
    if not gt_db:
        print(f"Error: No metadata for {db_folder_name}")
        return []

    print(f"Starting queries for DB {db_size}...")
    
    client = LabeledClient(APSI_PARAMS)
    results = []
    
    # 3. Run Queries
    # Use tqdm to show progress for queries
    for q in tqdm(precomputed_queries, desc=f"Queries for DB {db_size}", leave=False):
        tokens = q['tokens']
        if not tokens: continue
        
        try:
            # OPRF
            oprf_req = client.oprf_request(tokens)
            resp = requests.post(f"{server_url}/oprf", data=oprf_req, headers={'Content-Type': 'application/octet-stream'}, timeout=300)
            resp.raise_for_status()
            oprf_res = resp.content
            
            # Query
            query = client.build_query(oprf_res)
            resp = requests.post(f"{server_url}/query", data=query, headers={'Content-Type': 'application/octet-stream'}, timeout=300)
            resp.raise_for_status()
            query_res = resp.content
            
            # Decrypt
            matches = client.extract_result(query_res)
            
            # Ranking
            # matches is a dict {token: label}, we need to count labels
            if isinstance(matches, dict):
                counts = Counter(matches.values())
            else:
                counts = Counter(matches)
                
            ranked_results = [rid for rid, count in counts.most_common()]
            
            # DEBUG: Print info for the first query of the first DB
            if db_size == 1000 and q['id'] == "QUERY_0":
                tqdm.write(f"\n--- DEBUG QUERY_0 ---")
                tqdm.write(f"Query Labels: {q['labels']}")
                tqdm.write(f"Retrieved {len(ranked_results)} items: {ranked_results[:5]}...")
                for rid in ranked_results[:5]:
                    r_labels = gt_db.get(rid, "NOT_FOUND")
                    tqdm.write(f"  ID: {rid} -> Labels: {r_labels}")
                    if isinstance(r_labels, set):
                        tqdm.write(f"  Overlap: {q['labels'].intersection(r_labels)}")
                tqdm.write("---------------------\n")

            # Metrics
            m = calculate_metrics(ranked_results, q['labels'], gt_db, TOP_K_VALUES)
            
            row = {
                "DB_Size": db_size,
                "Query_ID": q['id'],
                "Labels": ";".join(q['labels']),
                "Retrieved_Count": len(ranked_results)
            }
            for k in TOP_K_VALUES:
                row[f"P@{k}"] = m[k]['p']
                row[f"R@{k}"] = m[k]['r']
                row[f"nDCG@{k}"] = m[k]['ndcg']
                row[f"LabelMatch@{k}"] = m[k]['label_match']
            
            results.append(row)
            
        except Exception as e:
            # print(f"Query failed: {e}")
            pass
            
    print(f"Finished DB {db_size}. Records: {len(results)}")
    return results

def get_query_images(num_queries=50):
    """
    Selects random images from valid.csv to use as queries.
    """
    if not DATASET_CSV.exists():
        print(f"Error: {DATASET_CSV} not found.")
        return []
        
    queries = []
    with open(DATASET_CSV, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    # Shuffle and pick
    import random
    random.shuffle(rows)
    
    LABELS = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
        "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
    ]
    
    count = 0
    for row in rows:
        path = _project_root / "dataset" / row['Path'].replace("CheXpert-v1.0-small/", "")
        if path.exists():
            # Extract labels
            labels = set()
            for lbl in LABELS:
                try:
                    if float(row.get(lbl, 0)) == 1.0:
                        labels.add(lbl)
                except: pass
            
            if not labels:
                if row.get("No Finding") == "1.0": labels.add("No Finding")
                else: labels.add("Unknown")
                
            queries.append({
                "path": str(path),
                "labels": labels,
                "id": f"QUERY_{count}"
            })
            count += 1
            if num_queries is not None and count >= num_queries:
                break
    return queries

def calculate_metrics(ranked_results, query_labels, ground_truth_db, k_list):
    """
    Calculates retrieval accuracy metrics: Precision@k, Recall@k, nDCG@k, and Label Match@k.

    --- DEFINITIONS & FORMULAS ---

    1. RELEVANCE (Sự liên quan):
       - A retrieved image is considered "Relevant" to the query image if they share AT LEAST ONE label.
       - Relevance(q, d) = 1 if (Labels(q) ∩ Labels(d)) ≠ ∅, else 0.
       - Note: "No Finding" is treated as a valid label.

    2. Precision@k (Độ chính xác tại k):
       - The proportion of relevant items in the top-k retrieved results.
       - Formula: P@k = (Number of Relevant Items in Top-k) / k

    3. Recall@k (Độ phủ tại k):
       - The proportion of relevant items retrieved in the top-k out of ALL relevant items in the database.
       - Formula: R@k = (Number of Relevant Items in Top-k) / (Total Relevant Items in Database)

    4. nDCG@k (Normalized Discounted Cumulative Gain):
       - Measures the quality of ranking, giving more weight to relevant items at the top.
       - DCG@k = Σ (Relevance(i) / log2(i + 1)) for i from 1 to k.
       - IDCG@k = Ideal DCG (if all relevant items were at the top).
       - Formula: nDCG@k = DCG@k / IDCG@k

    5. Label Match@k (Tỷ lệ khớp nhãn):
       - The average percentage of label overlap between the query and retrieved items.
       - For each retrieved item d: Match(d) = |Labels(q) ∩ Labels(d)| / |Labels(q)|
       - Formula: LabelMatch@k = (Σ Match(d) for d in Top-k) / k

    ------------------------------
    ranked_results: list of record_ids sorted by relevance (match count)
    query_labels: set of labels for the query image
    ground_truth_db: dict of {record_id: labels} for the whole DB
    """
    metrics = {}
    
    # 1. Identify ALL relevant items in DB (for Recall)
    # Relevant = shares at least one label (Jaccard > 0)
    # Note: "No Finding" vs "No Finding" is relevant.
    total_relevant_in_db = 0
    for rid, r_labels in ground_truth_db.items():
        if not query_labels.isdisjoint(r_labels):
            total_relevant_in_db += 1
            
    if total_relevant_in_db == 0:
        # Edge case: No relevant items in DB?
        return {k: {'p': 0.0, 'r': 0.0, 'ndcg': 0.0, 'label_match': 0.0} for k in k_list}

    # 2. Calculate metrics for each k
    for k in k_list:
        top_k_ids = ranked_results[:k]
        
        # Relevance vector (binary)
        relevance_binary = []
        for rid in top_k_ids:
            r_labels = ground_truth_db.get(rid, set())
            is_relevant = 1 if not query_labels.isdisjoint(r_labels) else 0
            relevance_binary.append(is_relevant)
            
        # Precision @ k
        relevant_retrieved = sum(relevance_binary)
        p_at_k = relevant_retrieved / k if k > 0 else 0
        
        # Recall @ k
        r_at_k = relevant_retrieved / total_relevant_in_db
        
        # nDCG @ k
        # DCG
        dcg = 0
        for i, rel in enumerate(relevance_binary):
            dcg += rel / np.log2(i + 2)
            
        # IDCG (Ideal DCG) - Best possible ordering
        # Imagine we retrieved min(k, total_relevant) relevant items at the top
        ideal_rel_count = min(k, total_relevant_in_db)
        ideal_vector = [1] * ideal_rel_count + [0] * (k - ideal_rel_count)
        idcg = 0
        for i, rel in enumerate(ideal_vector):
            idcg += rel / np.log2(i + 2)
            
        ndcg_at_k = dcg / idcg if idcg > 0 else 0
        
        # Label Match @ k (User requested: % of matching labels)
        # For each retrieved item, calculate |Intersection| / |Query Labels|
        # Then average over k
        label_match_sum = 0
        for rid in top_k_ids:
            r_labels = ground_truth_db.get(rid, set())
            intersection = len(query_labels.intersection(r_labels))
            denom = len(query_labels) if len(query_labels) > 0 else 1
            label_match_sum += intersection / denom
            
        avg_label_match = label_match_sum / k if k > 0 else 0
        
        metrics[k] = {
            'p': p_at_k,
            'r': r_at_k,
            'ndcg': ndcg_at_k,
            'label_match': avg_label_match
        }
        
    return metrics

def plot_accuracy():
    if not LOG_FILE.exists():
        print(f"Error: {LOG_FILE} not found.")
        return
        
    print(f"Reading data from {LOG_FILE}...")
    df = pd.read_csv(LOG_FILE)
    
    # Group by DB_Size and calculate mean for all metrics
    metric_cols = [c for c in df.columns if c.startswith(('P@', 'R@', 'nDCG@', 'LabelMatch@'))]
    grouped = df.groupby('DB_Size')[metric_cols].mean()
    
    db_sizes = grouped.index
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # We will plot metrics for k=10 as the representative metric
    k = 10
    
    # Check if columns exist before plotting
    if f'P@{k}' in grouped.columns:
        ax.plot(db_sizes, grouped[f'P@{k}'], marker='o', linestyle='-', linewidth=2, label=f'Precision@{k}')
        ax.plot(db_sizes, grouped[f'R@{k}'], marker='s', linestyle='--', linewidth=2, label=f'Recall@{k}')
        ax.plot(db_sizes, grouped[f'nDCG@{k}'], marker='^', linestyle='-.', linewidth=2, label=f'nDCG@{k}')
        # LabelMatch is often similar to Precision, but let's include it if distinct
        ax.plot(db_sizes, grouped[f'LabelMatch@{k}'], marker='x', linestyle=':', linewidth=2, label=f'Label Match@{k}')
    
    ax.set_xlabel('Database Size (Number of Patients)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Figure 4.3: Retrieval Accuracy vs Database Size (Top-{k})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    # Set x-axis ticks to match the DB sizes present
    ax.set_xticks(db_sizes)
    ax.set_xticklabels(db_sizes, rotation=45)
    
    # Set y-axis limits to 0-1 for normalized metrics
    ax.set_ylim(0, 1.05)
    
    fig.tight_layout()
    
    # Ensure directory exists
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Graph saved to {OUTPUT_IMAGE}")

def main():
    print("--- Setting up Accuracy Experiment (RQ4) - Sequential Mode ---")
    
    # Check if server is running
    try:
        requests.get("http://localhost:5000", timeout=5)
        print("Server is running on port 5000.")
    except requests.exceptions.ConnectionError:
        print("Error: Server is NOT running on port 5000. Please start it manually with: python3 server/server_app.py")
        return

    # 1. Find all data_xxx folders
    all_folders = sorted([f for f in os.listdir(SERVER_DATA_DIR) if f.startswith("data_") and (SERVER_DATA_DIR / f).is_dir()], 
                          key=lambda x: int(x.split('_')[1]))
    
    # Filter for specific DB sizes (approx every 1000)
    target_milestones = range(1000, 11000, 1000)
    data_folders = []
    
    for milestone in target_milestones:
        # Find folder with size closest to milestone
        best_folder = None
        min_diff = float('inf')
        
        for folder in all_folders:
            size = int(folder.split('_')[1])
            diff = abs(size - milestone)
            if diff < min_diff:
                min_diff = diff
                best_folder = folder
        
        if best_folder and best_folder not in data_folders:
            data_folders.append(best_folder)
            
    print(f"Selected {len(data_folders)} datasets for trend analysis: {data_folders}")
    
    # 2. Prepare Queries (Load & Tokenize ONCE)
    print("Loading and Tokenizing Queries...")
    raw_queries = get_query_images(num_queries=50) # Test with 50 images
    tokenizer = MedicalImageTokenizer()
    
    precomputed_queries = []
    for q in tqdm(raw_queries, desc="Tokenizing"):
        tokens = tokenizer.process(q['path'])
        if tokens:
            q['tokens'] = tokens # Attach tokens to query object
            precomputed_queries.append(q)
            
    print(f"Prepared {len(precomputed_queries)} valid queries.")
    
    # 3. Run Sequential Execution
    # Prepare CSV
    headers = ["DB_Size", "Query_ID", "Labels", "Retrieved_Count"]
    for k in TOP_K_VALUES:
        headers.extend([f"P@{k}", f"R@{k}", f"nDCG@{k}", f"LabelMatch@{k}"])
        
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for folder in tqdm(data_folders, desc="Testing DBs"):
            # Run experiment for this DB
            # We use a fixed port since we run sequentially
            results = run_experiment_for_db(folder, precomputed_queries)
            
            if results:
                writer.writerows(results)
                f.flush() # Ensure data is written to disk
            
            # Small delay to ensure port is released
            time.sleep(1)
            
    print(f"Results saved to {LOG_FILE}")
    plot_accuracy()

if __name__ == "__main__":
    main()
