# server/db_manager.py
from apsi import LabeledServer
from common.config import APSI_PARAMS, MAX_LABEL_LENGTH
import os

class APSIDatabase:
    def __init__(self, db_path=None):
        self.server = LabeledServer()
        if db_path is None:
            # Default to absolute path relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(base_dir, "data", "medical.db")
        else:
            self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        if os.path.exists(self.db_path) and os.path.getsize(self.db_path) > 0:
            print(f"[DB] Loading existing DB from {self.db_path}")
            try:
                self.server.load_db(self.db_path)
            except Exception as e:
                print(f"[DB] Load failed ({e}). Initializing new DB.")
                self._init_db()
        else:
            print("[DB] No valid DB found. Initializing new DB.")
            self._init_db()

    def _init_db(self):
        self.server.init_db(APSI_PARAMS, max_label_length=MAX_LABEL_LENGTH)

    def add_record(self, record_id: str, tokens: list[str]):
        # Deduplicate tokens to prevent duplicate (token, label) pairs
        tokens = list(set(tokens))
        items_with_labels = []
        for t in tokens:
            items_with_labels.append((t, record_id))
        
        try:
            self.server.add_items(items_with_labels)
        except ValueError as e:
            print(f"[DB] Error adding items: {e}")

    def add_batch(self, records_tokens: list[tuple]):
        """
        Efficiently add a batch of records.
        records_tokens: List of (record_id, tokens_list) tuples.
        """
        items_with_labels = []
        for rid, tokens in records_tokens:
            for t in tokens:
                items_with_labels.append((t, rid))
        
        if items_with_labels:
            try:
                self.server.add_items(items_with_labels)
            except ValueError as e:
                print(f"[DB] Error adding batch: {e}")

    def save(self):
        """Persist the database to disk."""
        try:
            self.server.save_db(self.db_path)
            print(f"[DB] Saved database to {self.db_path}")
        except Exception as e:
            print(f"[DB] Error saving database: {e}")

    def handle_oprf(self, request_bytes):
        return self.server.handle_oprf_request(request_bytes)

    def handle_query(self, query_bytes):
        return self.server.handle_query(query_bytes)