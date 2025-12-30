import os
import sys
import time
import json
import requests
import csv
from pathlib import Path
from tqdm import tqdm
import shutil
import multiprocessing
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
SERVER_URL = "http://localhost:5000"
SERVER_DATA_DIR = _project_root / "server"
LOG_FILE = _project_root / "experiment/result/experiment_rq1_results.csv"
TEST_IMAGE_PATH = _project_root / "dataset/train/patient00001/study1/view1_frontal.jpg"  # Use a fixed image for consistent testing
OUTPUT_IMAGE = _project_root / "experiment/result/figure_4.1_latency.png"

def setup_server_db(db_folder_name):
    """
    Swaps the active server DB with the specified checkpoint.
    """
    src_dir = SERVER_DATA_DIR / db_folder_name
    dst_dir = SERVER_DATA_DIR / "data"
    
    if not src_dir.exists():
        print(f"Error: Source DB {src_dir} does not exist.")
        return False
        
    # Clear destination
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir()
    
    # Copy files
    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dst_dir)
            
    print(f"-> Swapped DB to {db_folder_name}")
    return True

def trigger_server_reload():
    """
    Tries to tell the server to reload its DB.
    """
    try:
        resp = requests.post(f"{SERVER_URL}/admin/reload_db")
        if resp.status_code == 200:
            print("-> Server reloaded DB successfully.")
            return True
        else:
            print(f"-> Server reload failed: {resp.status_code}")
            return False
    except Exception:
        print("-> Could not contact server to reload. Is it running?")
        return False

def run_experiment_for_db(db_size, writer):
    print(f"\n=== Testing with DB Size: {db_size} ===")
    
    # 1. Measure T_token (Client side)
    t0 = time.time()
    tokenizer = MedicalImageTokenizer()
    tokens = tokenizer.process(str(TEST_IMAGE_PATH))
    t_token = time.time() - t0
    
    if not tokens:
        print("Error: No tokens generated.")
        return

    # 2. Measure T_client_enc (OPRF Req + Build Query)
    
    client = LabeledClient(APSI_PARAMS)
    
    # 2a. OPRF Request Generation
    t1 = time.time()
    oprf_req = client.oprf_request(tokens)
    t_oprf_req_gen = time.time() - t1
    
    # 2b. OPRF Round Trip (Network + Server OPRF)
    t2 = time.time()
    try:
        resp = requests.post(f"{SERVER_URL}/oprf", data=oprf_req, headers={'Content-Type': 'application/octet-stream'})
        resp.raise_for_status()
        oprf_response_content = resp.content
    except Exception as e:
        print(f"OPRF Failed: {e}")
        return
    t_oprf_roundtrip = time.time() - t2
    
    # 2c. Build Query (Client Enc)
    t3 = time.time()
    query = client.build_query(oprf_response_content)
    t_build_query = time.time() - t3
    
    t_client_enc = t_oprf_req_gen + t_build_query # Total client encryption effort
    
    # 3. Measure T_server_proc (Query Round Trip - Network)
    
    t4 = time.time()
    try:
        resp = requests.post(f"{SERVER_URL}/query", data=query, headers={'Content-Type': 'application/octet-stream'})
        resp.raise_for_status()
        query_response_content = resp.content
    except Exception as e:
        print(f"Query Failed: {e}")
        return
    t_query_roundtrip = time.time() - t4
    
    # 4. Measure T_dec (Client Decryption)
    t5 = time.time()
    try:
        # Just extract, don't care about content
        _ = client.extract_result(query_response_content)
    except Exception as e:
        print(f"Decryption Failed: {e}")
    t_dec = time.time() - t5
    
    # Log results
    t_total = t_token + t_client_enc + t_oprf_roundtrip + t_query_roundtrip + t_dec

    row = [
        db_size,
        f"{t_token:.4f}",
        f"{t_client_enc:.4f}",
        f"{t_oprf_roundtrip:.4f}",
        f"{t_query_roundtrip:.4f}", # Represents T_server_proc + T_net
        f"{t_dec:.4f}",
        f"{t_total:.4f}"
    ]
    
    writer.writerow(row)
    print(f"Results: Token={t_token:.3f}s, Enc={t_client_enc:.3f}s, OPRF_RTT={t_oprf_roundtrip:.3f}s, Query_RTT={t_query_roundtrip:.3f}s, Dec={t_dec:.3f}s, Total={t_total:.3f}s")

def plot_latency():
    if not LOG_FILE.exists():
        print(f"Error: {LOG_FILE} not found. Please run experiment first.")
        return

    # Read CSV
    try:
        df = pd.read_csv(LOG_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Calculate metrics for the graph
    # 1. Client Setup & Enc = T_token + T_client_enc
    # Note: We convert seconds to milliseconds (x1000) as the graph uses ms
    df['Client_Setup_Enc_ms'] = (df['T_token'] + df['T_client_enc']) * 1000
    
    # 2. Server PSI Eval = T_query_rtt
    # Note: T_query_rtt includes network time, but on localhost it's mostly Server Processing
    df['Server_PSI_Eval_ms'] = df['T_query_rtt'] * 1000
    
    # 3. Client Dec & Rank = T_dec
    df['Client_Dec_Rank_ms'] = df['T_dec'] * 1000
    
    # 4. Total Latency = T_total
    df['Total_Latency_ms'] = df['T_total'] * 1000

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot lines with markers
    plt.plot(df['DB_Size'], df['Client_Setup_Enc_ms'], marker='o', label='Client Setup & Enc (ms)')
    plt.plot(df['DB_Size'], df['Server_PSI_Eval_ms'], marker='o', label='Server PSI Eval (ms)')
    plt.plot(df['DB_Size'], df['Client_Dec_Rank_ms'], marker='o', label='Client Dec & Rank (ms)')
    plt.plot(df['DB_Size'], df['Total_Latency_ms'], marker='o', linewidth=3, label='Total Latency (ms)')

    # Formatting
    plt.xscale('log') # Logarithmic scale for X-axis (Database Size)
    plt.xlabel('Database Size (Number of Images)', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.title('Figure 4.1: Latency vs. Database Size', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    
    # Save and Show
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Graph saved to {OUTPUT_IMAGE}")

def main():
    # 1. Find all data_xxx folders
    data_folders = sorted([f for f in os.listdir(SERVER_DATA_DIR) if f.startswith("data_") and (SERVER_DATA_DIR / f).is_dir()], 
                          key=lambda x: int(x.split('_')[1]))
    
    print(f"Found {len(data_folders)} datasets to test: {data_folders}")
    
    # Prepare CSV
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["DB_Size", "T_token", "T_client_enc", "T_oprf_rtt", "T_query_rtt", "T_dec", "T_total"])
        
        for folder in data_folders:
            db_size = int(folder.split('_')[1])
            
            # Swap DB
            if setup_server_db(folder):
                # Trigger Reload
                if trigger_server_reload():
                    # Run Test
                    run_experiment_for_db(db_size, writer)
                    f.flush() # Ensure write
                else:
                    print("Skipping due to reload failure.")
            
            time.sleep(1) # Cool down
            
    # Plot results
    plot_latency()

if __name__ == "__main__":
    main()
