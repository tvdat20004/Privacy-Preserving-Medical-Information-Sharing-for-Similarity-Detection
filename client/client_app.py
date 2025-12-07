# client/client_app.py
import os
import sys
from pathlib import Path

# Add project root to Python path for imports (must be before any project imports)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import requests
from collections import Counter
from apsi import LabeledClient
from common.config import APSI_PARAMS
from client.image.tokenizer import MedicalImageTokenizer

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:5000")

def run_client(image_path, top_k=3):
    # 1. Tokenize
    print(f"[Client] Tokenizing {image_path}...")
    tokenizer = MedicalImageTokenizer()
    tokens = tokenizer.process(image_path)
    
    if not tokens:
        print("[Client] Error: No tokens generated.")
        return []

    # 2. PSI Protocol
    print(f"[Client] Running PSI with {len(tokens)} tokens...")
    client = LabeledClient(APSI_PARAMS)
    
    try:
        # OPRF
        oprf_req = client.oprf_request(tokens)
        resp = requests.post(f"{SERVER_URL}/oprf", data=oprf_req, headers={'Content-Type': 'application/octet-stream'})
        resp.raise_for_status()
        
        # Query
        query = client.build_query(resp.content)
        resp = requests.post(f"{SERVER_URL}/query", data=query, headers={'Content-Type': 'application/octet-stream'})
        resp.raise_for_status()
        
        # Extract
        result = client.extract_result(resp.content)
    except Exception as e:
        print(f"[Client] Protocol connection error: {e}")
        return []

    # 3. Aggregate Results
    if not result:
        print("[Client] No matching records found.")
        return []

    print(f"[Client] Found {len(result)} matching tokens.")
    
    # Count matches per Patient ID
    counts = Counter(result.values())
    
    # Get top-k unique patient IDs
    top_matches = counts.most_common(top_k)
    
    final_output = []
    
    print("[Client] Fetching metadata for top matches...")
    for record_id, score in top_matches:
        try:
            # Fetch metadata
            meta_resp = requests.get(f"{SERVER_URL}/metadata/{record_id}")
            if meta_resp.status_code == 200:
                metadata = meta_resp.json()
            else:
                metadata = {"error": "Metadata not found on server"}

            # Format the JSON object
            match_obj = {
                "record_id": record_id,
                "score": score,
                "metadata": metadata
            }
            final_output.append(match_obj)
            
        except Exception as e:
            print(f"[Client] Error fetching {record_id}: {e}")

    return final_output

if __name__ == "__main__":
    img_path = "./image/dummy_xray_A.jpg"
    
    # Ensure dummy image exists
    if not os.path.exists(img_path):
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        from PIL import Image
        Image.new('RGB', (100, 100), color='white').save(img_path)

    matches = run_client(img_path, top_k=3)
    
    # Print final JSON result
    print("\n--- Final Result (JSON) ---")
    print(json.dumps(matches, indent=1))