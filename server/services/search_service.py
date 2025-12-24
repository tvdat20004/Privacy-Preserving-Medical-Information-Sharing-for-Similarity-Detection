import os
import tempfile
import json
import time
from collections import Counter
from apsi import LabeledClient
from common.config import APSI_PARAMS
from client.image.tokenizer import MedicalImageTokenizer

class SearchService:
    def __init__(self, record_manager):
        self.record_manager = record_manager

    def process_image_search(self, image_file, top_k=10):
        # Save to a temp file so the tokenizer can read by path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.filename)[1] or '.jpg') as tmp:
            image_file.save(tmp.name)
            temp_path = tmp.name

        try:
            tokenizer = MedicalImageTokenizer()
            tokens = tokenizer.process(temp_path)
        except Exception as e:
            os.unlink(temp_path)
            raise Exception(f"Tokenization error: {e}")

        os.unlink(temp_path)

        if not tokens:
            raise Exception("No tokens generated from image")

        return self._run_psi(tokens, top_k)

    def process_token_search(self, tokens, top_k=5):
        if not tokens or not isinstance(tokens, list):
            raise Exception("Invalid tokens provided")
        
        # Ensure tokens are strings
        tokens = [str(t) for t in tokens]
        
        return self._run_psi(tokens, top_k)

    def encode_image(self, image_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.filename)[1] or '.jpg') as tmp:
            image_file.save(tmp.name)
            temp_path = tmp.name

        try:
            tokenizer = MedicalImageTokenizer()
            tokens = tokenizer.process(temp_path)
        except Exception as e:
            os.unlink(temp_path)
            raise Exception(f"Tokenization error: {e}")

        os.unlink(temp_path)

        if not tokens:
            raise Exception("No tokens generated from image")
            
        return tokens

    def _run_psi(self, tokens, top_k):
        # Deduplicate tokens to avoid "duplicate positive match" error in APSI
        # Ensure all tokens are strings before deduplication
        tokens = sorted(list(set(str(t) for t in tokens)))

        # Run PSI locally via in-process server/db
        try:
            client = LabeledClient(APSI_PARAMS)
            
            # 1. OPRF Request
            oprf_req = client.oprf_request(tokens)
            
            # 2. Server OPRF Response
            oprf_resp = self.record_manager.handle_oprf(oprf_req)
            
            # 3. Build Query
            query = client.build_query(oprf_resp)
            
            # 4. Server Query Response
            start_time = time.time()
            query_resp = self.record_manager.handle_query(query)
            elapsed_time = time.time() - start_time
            print(f"[SearchService] PSI Query processed in {elapsed_time:.4f} seconds")
            
            # 5. Extract Result
            # Note: extract_result can fail if there are duplicate matches for the same item
            # This usually happens if the DB has duplicate (token, label) pairs
            # or if the client sends duplicate tokens (which we handled above).
            # However, if multiple DB items share the same token AND label, it might cause issues depending on APSI version.
            # But here, we are doing labeled PSI where label is the Record ID.
            # So (token, record_id) should be unique.
            result = client.extract_result(query_resp)
        except RuntimeError as e:
            if "duplicate positive match" in str(e):
                print(f"[PSI ERROR] Duplicate match detected. This implies database corruption or collision.")
                # Fallback: Try to process without crashing, or return empty
                return []
            raise Exception(f"PSI Protocol error: {e}")
        except Exception as e:
            raise Exception(f"PSI Protocol error: {e}")

        if not result:
            return []

        # Count matches per record
        counts = Counter(result.values())
        top_matches = counts.most_common(top_k)

        payload = []
        for record_id, score in top_matches:
            metadata = self.record_manager.get_metadata(record_id) or {"note": "Chưa có metadata"}
            payload.append({
                "record_id": record_id,
                "score": score,
                "metadata": metadata
            })

        return payload
