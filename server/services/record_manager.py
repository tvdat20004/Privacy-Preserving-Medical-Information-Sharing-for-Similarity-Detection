import os
import json
import uuid
import random
from pathlib import Path
from common.config import APSI_PARAMS, MAX_LABEL_LENGTH
from server.db_manager import APSIDatabase
from client.image.tokenizer import MedicalImageTokenizer

class MedicalRecordManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = self.project_root / "server" / "data"
        self.metadata_path = self.data_dir / "metadata.json"
        self.tokens_path = self.data_dir / "tokens.json"
        self.db_path = self.data_dir / "medical.db"
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        print("--- Initializing Medical Record Manager ---")
        self.db = APSIDatabase(db_path=str(self.db_path))
        
        self.metadata_store = self._load_metadata()
        self.tokens_store = self._load_tokens_store()
        print(f"[Manager] Loaded {len(self.metadata_store)} metadata records.")
        
        # Check if DB exists, but DO NOT auto-index anymore
        if not os.path.exists(self.db_path) or os.path.getsize(self.db_path) == 0:
            print("[Manager] Warning: Database is empty. Please run 'generate_db.py' to populate it.")
        else:
            print("[Manager] Database loaded successfully.")

    def _load_metadata(self):
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Manager] Error loading metadata: {e}")
        return {}

    def _load_tokens_store(self):
        if self.tokens_path.exists():
            try:
                with open(self.tokens_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Manager] Error loading tokens store: {e}")
        return {}

    def _save_metadata(self):
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata_store, f, indent=2)
            print(f"[Manager] Metadata saved to {self.metadata_path}")
        except Exception as e:
            print(f"[Manager] Error saving metadata: {e}")

    def _save_tokens_store(self):
        try:
            with open(self.tokens_path, "w") as f:
                json.dump(self.tokens_store, f)
            print(f"[Manager] Tokens saved to {self.tokens_path}")
        except Exception as e:
            print(f"[Manager] Error saving tokens: {e}")

    # _run_offline_indexing and related methods removed as they are now in generate_db.py

    def get_metadata(self, record_id):
        return self.metadata_store.get(record_id)

    def get_all_metadata(self):
        return [
            {"record_id": rid, **meta} if isinstance(meta, dict) else {"record_id": rid, "meta": meta}
            for rid, meta in self.metadata_store.items()
        ]

    def add_patient_record(self, image_path, diagnosis_list, age, name):
        try:
            tokenizer = MedicalImageTokenizer()
            tokens = tokenizer.process(image_path)
        except Exception as e:
            raise Exception(f"Tokenization failed: {e}")

        if not tokens:
            raise Exception("No tokens generated from image")

        record_id = f"PATIENT_{uuid.uuid4().hex[:8].upper()}"
        
        try:
            self.db.add_record(record_id, tokens)
            self.db.save()
            
            self.metadata_store[record_id] = {
                "record_id": record_id,
                "diagnosis": diagnosis_list,
                "age": age,
                "name": name
            }
            self.tokens_store[record_id] = tokens
            
            self._save_metadata()
            self._save_tokens_store()
            
            return record_id
        except Exception as e:
            raise Exception(f"Database save failed: {e}")

    def delete_patient_record(self, record_id):
        if record_id not in self.metadata_store:
            return False

        # Remove from stores
        self.tokens_store.pop(record_id, None)
        self.metadata_store.pop(record_id, None)
        self._save_metadata()
        self._save_tokens_store()

        # Rebuild DB from tokens
        try:
            # Re-initialize DB (clears it)
            self.db.server.init_db(APSI_PARAMS, max_label_length=MAX_LABEL_LENGTH)
            for rid, toks in self.tokens_store.items():
                self.db.add_record(rid, toks)
            self.db.save()
            return True
        except Exception as e:
            raise Exception(f"DB Rebuild failed: {e}")

    def handle_oprf(self, request_data):
        return self.db.handle_oprf(request_data)

    def handle_query(self, query_data):
        return self.db.handle_query(query_data)
