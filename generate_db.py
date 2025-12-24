import os
import sys
import csv
import uuid
import random
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Add project root to Python path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.config import APSI_PARAMS, MAX_LABEL_LENGTH
from server.db_manager import APSIDatabase
from client.image.tokenizer import MedicalImageTokenizer

class MedicalDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform
        
    def __len__(self):
        return len(self.rows)
        
    def __getitem__(self, idx):
        row = self.rows[idx]
        path = row['_local_path']
        try:
            # Open image
            img = Image.open(path).convert('RGB')
            # Apply transform
            if self.transform:
                img = self.transform(img)
            return img, idx
        except Exception as e:
            # print(f"Error loading {path}: {e}")
            return None, idx

def collate_fn(batch):
    # Filter out failed loads
    valid_batch = [b for b in batch if b[0] is not None]
    if not valid_batch:
        return None, []
    
    imgs, idxs = zip(*valid_batch)
    return torch.stack(imgs), idxs

def generate_database(start_index, end_index, append=False, checkpoint_interval=None):
    mode_str = "Appending to" if append else "Generating new"
    print(f"--- {mode_str} Database with samples {start_index}-{end_index} (Optimized) ---")
    if checkpoint_interval:
        print(f"--- Checkpoint interval: {checkpoint_interval} ---")
    
    # Setup paths
    data_dir = _project_root / "server" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
     
    # Final paths
    db_path = data_dir / "medical.db"
    metadata_path = data_dir / "metadata.json"
    tokens_path = data_dir / "tokens.json"

    # Initialize DB
    db = APSIDatabase(db_path=str(db_path))
    
    metadata_store = {}
    tokens_store = {}

    if append:
        # Load existing data if appending
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata_store = json.load(f)
        if tokens_path.exists():
            with open(tokens_path, "r") as f:
                tokens_store = json.load(f)
        print(f"Loaded existing {len(metadata_store)} records.")
    else:
        # Clear old data if not appending (re-init DB)
        # Note: APSIDatabase constructor loads existing DB if found, 
        # so we need to explicitly clear it if we want a fresh start.
        if db_path.exists():
             # Re-initialize the underlying APSI server to clear data
             db._init_db() 

    # Collect existing paths to avoid duplicates
    existing_paths = set()
    if metadata_store:
        for rec in metadata_store.values():
            if 'original_path' in rec:
                existing_paths.add(rec['original_path'])
    if existing_paths:
        print(f"Found {len(existing_paths)} existing images. Will skip these.")
    
    # Load Dataset
    dataset_root = _project_root / "dataset"
    train_csv = dataset_root / "train.csv"
    
    if not train_csv.exists():
        print(f"Error: Dataset not found at {train_csv}")
        return

    # Read all rows
    print("Reading CSV...")
    with open(train_csv, 'r') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    
    print(f"Total rows in CSV: {len(all_rows)}")
    
    # Validate range
    if start_index < 0: start_index = 0
    if end_index > len(all_rows): end_index = len(all_rows)
    
    target_rows = all_rows[start_index:end_index]
    
    selected_rows = []
    print(f"Selecting valid samples from range [{start_index}:{end_index}]...")
    
    for row in tqdm(target_rows, desc="Checking files"):
        # Fix path
        rel_path = row['Path']
        if rel_path.startswith("CheXpert-v1.0-small/"):
            rel_path = rel_path.replace("CheXpert-v1.0-small/", "")
        
        # Skip if already in DB
        if rel_path in existing_paths:
            continue

        img_path = dataset_root / rel_path
        
        # Check existence
        if img_path.exists():
            row['_local_path'] = str(img_path)
            row['_rel_path'] = rel_path
            selected_rows.append(row)
            
    if not selected_rows:
        print(f"Warning: No valid images found in range {start_index}-{end_index}.")
        
    # Initialize Tokenizer
    tok = MedicalImageTokenizer()
    
    # Create Dataset and DataLoader
    # num_workers=8 for parallel loading/preprocessing
    # pin_memory=True for faster transfer to GPU
    dataset = MedicalDataset(selected_rows, tok.transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Labels definition
    LABELS = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
        "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
    ]
    
    print("Tokenizing and Indexing (DataLoader + GPU)...")
    
    processed_count = 0
    next_checkpoint = checkpoint_interval if checkpoint_interval else float('inf')

    for batch_tensors, batch_idxs in tqdm(dataloader, desc="Processing Batches"):
        if batch_tensors is None:
            continue
            
        # Process batch on GPU
        # process_tensors handles moving to device
        batch_tokens_list = tok.process_tensors(batch_tensors)
        
        db_batch = []
        
        for i, tokens in enumerate(batch_tokens_list):
            idx = batch_idxs[i]
            row = selected_rows[idx]
            
            record_id = f"PATIENT_{uuid.uuid4().hex[:8].upper()}"
            
            # Extract Diagnosis
            diagnosis = []
            for lbl in LABELS:
                val = row.get(lbl, "")
                try:
                    if float(val) == 1.0:
                        diagnosis.append(lbl)
                except ValueError:
                    pass
            if not diagnosis:
                diagnosis = ["No Finding"] if row.get("No Finding") == "1.0" else ["Unknown"]
                
            record = {
                "record_id": record_id,
                "age": row.get('Age', 'N/A'),
                "sex": row.get('Sex', 'N/A'),
                "diagnosis": diagnosis,
                "original_path": row['_rel_path']
            }
            
            metadata_store[record_id] = record
            tokens_store[record_id] = tokens
            
            # Deduplicate tokens to prevent "duplicate positive match" error
            unique_tokens = list(set(tokens))
            db_batch.append((record_id, unique_tokens))
            
        # Add to DB
        if db_batch:
            db.add_batch(db_batch)
            processed_count += len(db_batch)

        # Checkpoint Logic
        if checkpoint_interval and processed_count >= next_checkpoint:
            total_records = len(metadata_store)
            print(f"\n[Checkpoint] Reached {processed_count} processed items. Saving snapshot to data_{total_records}...")
            
            # Save main DB first
            db.save()
            
            # Create checkpoint directory
            ckpt_dir = _project_root / "server" / f"data_{total_records}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy DB file
            try:
                shutil.copy2(db_path, ckpt_dir / "medical.db")
            except Exception as e:
                print(f"[Checkpoint] Error copying DB: {e}")
                
            # Save JSONs
            with open(ckpt_dir / "metadata.json", "w") as f:
                json.dump(metadata_store, f, indent=2)
            with open(ckpt_dir / "tokens.json", "w") as f:
                json.dump(tokens_store, f)
                
            print(f"[Checkpoint] Snapshot saved.")
            next_checkpoint += checkpoint_interval
        
    # Save Stores
    print("Saving Database...")
    db.save()
    with open(metadata_path, "w") as f:
        json.dump(metadata_store, f, indent=2)
    with open(tokens_path, "w") as f:
        json.dump(tokens_store, f)
        
    print(f"--- Done! Database now has {len(metadata_store)} records. ---")

import argparse

if __name__ == "__main__":
    # Enable multiprocessing support for PyTorch
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description="Generate or update the medical image database.")
    parser.add_argument("--mode", choices=["append", "replace"], required=True, help="Mode: 'append' to add to existing DB, 'replace' to create new DB.")
    parser.add_argument("--start", type=int, default=0, help="Start index in dataset.")
    parser.add_argument("--end", type=int, required=True, help="End index in dataset.")
    parser.add_argument("--checkpoint", type=int, default=None, help="Save checkpoint every N samples.")
    
    args = parser.parse_args()
    
    is_append = (args.mode == "append")
    
    generate_database(start_index=args.start, end_index=args.end, append=is_append, checkpoint_interval=args.checkpoint)


