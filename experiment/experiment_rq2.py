import os
import sys
import time
import json
import requests
import csv
from pathlib import Path
import shutil
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
LOG_FILE = _project_root / "experiment/result/experiment_rq2_results.csv"
TEST_IMAGE_PATH = _project_root / "dataset/train/patient00001/study1/view1_frontal.jpg"  # Use a fixed image for consistent testing
OUTPUT_IMAGE = _project_root / "experiment/result/figure_4.2_communication_cost.png"

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
    print(f"\n=== Measuring Data Transfer with DB Size: {db_size} ===")
    
    # 1. Tokenize
    tokenizer = MedicalImageTokenizer()
    tokens = tokenizer.process(str(TEST_IMAGE_PATH))
    
    if not tokens:
        print("Error: No tokens generated.")
        return

    client = LabeledClient(APSI_PARAMS)
    
    # 2. OPRF Request
    oprf_req = client.oprf_request(tokens)
    size_oprf_req = len(oprf_req)
    
    # 3. OPRF Response
    try:
        resp = requests.post(f"{SERVER_URL}/oprf", data=oprf_req, headers={'Content-Type': 'application/octet-stream'})
        resp.raise_for_status()
        oprf_response_content = resp.content
        size_oprf_res = len(oprf_response_content)
    except Exception as e:
        print(f"OPRF Failed: {e}")
        return
    
    # 4. Query Request
    query = client.build_query(oprf_response_content)
    size_query_req = len(query)
    
    # 5. Query Response
    try:
        resp = requests.post(f"{SERVER_URL}/query", data=query, headers={'Content-Type': 'application/octet-stream'})
        resp.raise_for_status()
        query_response_content = resp.content
        size_query_res = len(query_response_content)
    except Exception as e:
        print(f"Query Failed: {e}")
        return
    
    # Calculate Total
    total_size = size_oprf_req + size_oprf_res + size_query_req + size_query_res
    
    # Log results (Sizes in Bytes)
    row = [
        db_size,
        size_oprf_req,
        size_oprf_res,
        size_query_req,
        size_query_res,
        total_size
    ]
    
    writer.writerow(row)
    print(f"Sizes (Bytes): OPRF_Req={size_oprf_req}, OPRF_Res={size_oprf_res}, Query_Req={size_query_req}, Query_Res={size_query_res}, Total={total_size}")

def plot_communication_cost():
    if not LOG_FILE.exists():
        print(f"Error: {LOG_FILE} not found. Please run experiment first.")
        return

    # Read CSV
    try:
        df = pd.read_csv(LOG_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Convert Bytes to Megabytes (MB)
    # 1 MB = 1024 * 1024 Bytes
    BYTES_TO_MB = 1 / (1024 * 1024)

    df['Query_Req_MB'] = df['Size_Query_Req'] * BYTES_TO_MB
    df['Query_Res_MB'] = df['Size_Query_Res'] * BYTES_TO_MB
    df['Total_Size_MB'] = df['Total_Size_Bytes'] * BYTES_TO_MB
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot lines with markers
    plt.plot(df['DB_Size'], df['Query_Req_MB'], marker='s', label='Query Request (Client -> Server)')
    plt.plot(df['DB_Size'], df['Query_Res_MB'], marker='^', label='Query Response (Server -> Client)')
    plt.plot(df['DB_Size'], df['Total_Size_MB'], marker='o', linewidth=3, label='Total Data Transfer')

    # Formatting
    plt.xscale('log') # Logarithmic scale for X-axis to match RQ2 style
    plt.xlabel('Database Size (Number of Images)', fontsize=12)
    plt.ylabel('Communication Cost (MB)', fontsize=12)
    plt.title('Figure 4.2: Communication Cost vs. Database Size', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    
    # Add text annotation for OPRF (since it's too small to plot)
    avg_oprf = (df['Size_OPRF_Req'].mean() + df['Size_OPRF_Res'].mean()) / 1024
    plt.annotate(f'Note: OPRF traffic is negligible (~{avg_oprf:.2f} KB)', 
                 xy=(0.02, 0.02), xycoords='axes fraction', fontsize=9, style='italic')

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
        writer.writerow(["DB_Size", "Size_OPRF_Req", "Size_OPRF_Res", "Size_Query_Req", "Size_Query_Res", "Total_Size_Bytes"])
        
        for folder in data_folders:
            db_size = int(folder.split('_')[1])
            
            # Swap DB
            if setup_server_db(folder):
                # Trigger Reload
                if trigger_server_reload():
                    run_experiment_for_db(db_size, writer)
                    f.flush()
                else:
                    print("Skipping due to reload failure.")
            
            time.sleep(1)
            
    # Plot results
    plot_communication_cost()

if __name__ == "__main__":
    main()
