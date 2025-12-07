# server/server_app.py
import os
import sys
import uuid
import random
import json
from pathlib import Path

# Add project root to Python path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import tempfile
from common.config import APSI_PARAMS, MAX_LABEL_LENGTH
from flask import Flask, request, jsonify, Response, render_template, redirect, session, url_for
from server.db_manager import APSIDatabase
from client.image.tokenizer import MedicalImageTokenizer
from apsi import LabeledClient

TEMPLATE_ROOT = _project_root / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_ROOT))
app.secret_key = os.environ.get("APP_SECRET_KEY", "dev-secret")

print("--- Initializing Server and Database ---")
db = APSIDatabase(db_path="server/data/medical.db")
app.db = db

# --- METADATA PERSISTENCE SETUP ---
METADATA_PATH = Path("server/data/metadata.json")
TOKENS_PATH = Path("server/data/tokens.json")

def load_metadata():
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Server] Error loading metadata: {e}")
    return {}


def load_tokens_store():
    if TOKENS_PATH.exists():
        try:
            with open(TOKENS_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Server] Error loading tokens store: {e}")
    return {}

def save_metadata(store):
    try:
        with open(METADATA_PATH, "w") as f:
            json.dump(store, f, indent=2)
        print(f"[Server] Metadata saved to {METADATA_PATH}")
    except Exception as e:
        print(f"[Server] Error saving metadata: {e}")


def save_tokens_store(store):
    try:
        with open(TOKENS_PATH, "w") as f:
            json.dump(store, f)
        print(f"[Server] Tokens saved to {TOKENS_PATH}")
    except Exception as e:
        print(f"[Server] Error saving tokens: {e}")

# Load metadata into memory on startup
app.metadata_store = load_metadata()
app.tokens_store = load_tokens_store()
print(f"[Server] Loaded {len(app.metadata_store)} metadata records.")

# --- OFFLINE PHASE: INDEXING ---
if not os.path.exists(db.db_path) or os.path.getsize(db.db_path) == 0:
    print("Performing Offline Indexing...")
    tok = MedicalImageTokenizer()
    
    # Scan test_data directory
    test_data_dir = Path(_project_root) / "tests" / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in test_data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    # Indexing Logic
    diagnoses = ["Pneumonia", "Normal", "Atelectasis", "Cardiomegaly", "Effusion"]
    
    count = 0
    for img_path in sorted(image_files):
        # Create a deterministic ID for testing (or random)
        # Using filename hash or just filename helps debugging
        record_id = f"PATIENT_{uuid.uuid4().hex[:8].upper()}"
        
        try:
            tokens = tok.process(str(img_path))
            if tokens:
                db.add_record(record_id, tokens)
                
                # Populate metadata
                app.metadata_store[record_id] = {
                    "record_id": record_id,
                    "age": random.randint(18, 85),
                    "diagnosis": random.choice(diagnoses)
                }
                app.tokens_store[record_id] = tokens
                count += 1
                print(f"  ✓ Indexed {record_id} ({img_path.name})")
        except Exception as e:
            print(f"  ✗ Error {img_path.name}: {e}")
            
    # Save BOTH the DB and the Metadata
    db.save()
    save_metadata(app.metadata_store)
    save_tokens_store(app.tokens_store)
    print(f"--- Indexing Complete: {count} records ---")
else:
    print("Database already exists. Skipping indexing.")

@app.route('/')
def root_redirect():
    return redirect('/app')


def require_admin():
    return session.get("is_admin") is True


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'GET':
        return render_template('login.html')

    data = request.form or request.get_json(silent=True) or {}
    username = data.get('username')
    password = data.get('password')
    if username == 'admin' and password == '12345':
        session['is_admin'] = True
        return jsonify({"ok": True, "redirect": url_for('admin_home')})
    return jsonify({"error": "Sai tài khoản hoặc mật khẩu"}), 401


@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    session.pop('is_admin', None)
    return jsonify({"ok": True})


# ---------------- PSI API (shared) -----------------
@app.route('/oprf', methods=['POST'])
def oprf():
    return Response(app.db.handle_oprf(request.get_data()), mimetype='application/octet-stream')


@app.route('/query', methods=['POST'])
def query():
    return Response(app.db.handle_query(request.get_data()), mimetype='application/octet-stream')


@app.route('/metadata/<record_id>', methods=['GET'])
def get_metadata(record_id):
    meta = app.metadata_store.get(record_id)
    if meta:
        return jsonify(meta)
    else:
        return jsonify({"error": "Record ID found in PSI but missing in Metadata store"}), 404


# ---------------- Client UI -----------------
@app.route('/app')
def ui_home():
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
@app.route('/ui/search', methods=['POST'])  # backward compatibility
def ui_search():
    if 'image' not in request.files:
        return jsonify({"error": "Thiếu file ảnh (field name: image)"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "File ảnh không hợp lệ"}), 400

    # Save to a temp file so the tokenizer can read by path
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or '.jpg') as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        tokenizer = MedicalImageTokenizer()
        tokens = tokenizer.process(temp_path)
    except Exception as e:
        os.unlink(temp_path)
        return jsonify({"error": f"Lỗi tokenize ảnh: {e}"}), 500

    os.unlink(temp_path)

    if not tokens:
        return jsonify({"error": "Không tạo được token từ ảnh"}), 400

    # Run PSI locally via in-process server/db
    try:
        client = LabeledClient(APSI_PARAMS)
        oprf_req = client.oprf_request(tokens)
        oprf_resp = app.db.handle_oprf(oprf_req)
        query = client.build_query(oprf_resp)
        query_resp = app.db.handle_query(query)
        result = client.extract_result(query_resp)
    except Exception as e:
        return jsonify({"error": f"Lỗi khi chạy PSI: {e}"}), 500

    if not result:
        return jsonify({"matches": [], "message": "Không tìm thấy bản ghi phù hợp"})

    # Count matches per record
    from collections import Counter
    counts = Counter(result.values())
    top_matches = counts.most_common(5)

    payload = []
    for record_id, score in top_matches:
        metadata = app.metadata_store.get(record_id, {"note": "Chưa có metadata"})
        payload.append({
            "record_id": record_id,
            "score": score,
            "metadata": metadata
        })

    return jsonify({"matches": payload})


def _run_psi_with_tokens(tokens, top_k=5):
    """Shared PSI execution given precomputed tokens."""
    client = LabeledClient(APSI_PARAMS)
    oprf_req = client.oprf_request(tokens)
    oprf_resp = app.db.handle_oprf(oprf_req)
    query = client.build_query(oprf_resp)
    query_resp = app.db.handle_query(query)
    result = client.extract_result(query_resp)
    if not result:
        return []
    from collections import Counter
    counts = Counter(result.values())
    top_matches = counts.most_common(top_k)
    payload = []
    for record_id, score in top_matches:
        metadata = app.metadata_store.get(record_id, {"note": "Chưa có metadata"})
        payload.append({
            "record_id": record_id,
            "score": score,
            "metadata": metadata
        })
    return payload


@app.route('/api/encode', methods=['POST'])
def api_encode():
    if 'image' not in request.files:
        return jsonify({"error": "Thiếu file ảnh (field name: image)"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "File ảnh không hợp lệ"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or '.jpg') as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        tokenizer = MedicalImageTokenizer()
        tokens = tokenizer.process(temp_path)
    except Exception as e:
        os.unlink(temp_path)
        return jsonify({"error": f"Lỗi tokenize ảnh: {e}"}), 500

    os.unlink(temp_path)

    if not tokens:
        return jsonify({"error": "Không tạo được token từ ảnh"}), 400

    payload = {"tokens": tokens}
    return jsonify({"count": len(tokens), "enc": payload})


@app.route('/api/search_tokens', methods=['POST'])
def api_search_tokens():
    tokens = None

    if 'enc' in request.files:
        enc_file = request.files['enc']
        try:
            data = json.load(enc_file)
            tokens = data.get('tokens')
        except Exception as e:
            return jsonify({"error": f"Không đọc được file enc: {e}"}), 400
    else:
        try:
            data = request.get_json(force=True)
            tokens = data.get('tokens') if data else None
        except Exception:
            tokens = None

    if not tokens or not isinstance(tokens, list):
        return jsonify({"error": "Thiếu tokens"}), 400

    try:
        payload = _run_psi_with_tokens(tokens)
    except Exception as e:
        return jsonify({"error": f"Lỗi khi chạy PSI: {e}"}), 500

    if not payload:
        return jsonify({"matches": [], "message": "Không tìm thấy bản ghi phù hợp"})

    return jsonify({"matches": payload})


# ---------------- Admin UI -----------------
@app.route('/admin')
def admin_home():
    if not require_admin():
        return redirect(url_for('admin_login'))
    return render_template('admin.html')


@app.route('/admin/metadata', methods=['GET'])
def admin_metadata():
    if not require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    records = [
        {"record_id": rid, **meta} if isinstance(meta, dict) else {"record_id": rid, "meta": meta}
        for rid, meta in app.metadata_store.items()
    ]
    return jsonify({"count": len(records), "records": records})


@app.route('/admin/upload', methods=['POST'])
def admin_upload():
    if not require_admin():
        return jsonify({"error": "Unauthorized"}), 401

    if 'image' not in request.files:
        return jsonify({"error": "Thiếu file ảnh (field name: image)"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "File ảnh không hợp lệ"}), 400

    diagnosis = request.form.get('diagnosis') or ''
    age_raw = request.form.get('age') or ''
    name = request.form.get('name') or ''
    try:
        age = int(age_raw)
    except Exception:
        age = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or '.jpg') as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        tokenizer = MedicalImageTokenizer()
        tokens = tokenizer.process(temp_path)
    except Exception as e:
        os.unlink(temp_path)
        return jsonify({"error": f"Lỗi tokenize ảnh: {e}"}), 500

    os.unlink(temp_path)

    if not tokens:
        return jsonify({"error": "Không tạo được token từ ảnh"}), 400

    record_id = f"PATIENT_{uuid.uuid4().hex[:8].upper()}"
    try:
        app.db.add_record(record_id, tokens)
        app.db.save()
        app.metadata_store[record_id] = {
            "record_id": record_id,
            "diagnosis": diagnosis or "N/A",
            "age": age,
            "name": name
        }
        app.tokens_store[record_id] = tokens
        save_metadata(app.metadata_store)
        save_tokens_store(app.tokens_store)
    except Exception as e:
        return jsonify({"error": f"Lỗi lưu DB: {e}"}), 500

    return jsonify({"ok": True, "record_id": record_id})


@app.route('/admin/delete', methods=['POST'])
def admin_delete():
    if not require_admin():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    record_id = data.get('record_id')
    if not record_id:
        return jsonify({"error": "Thiếu record_id"}), 400

    if record_id not in app.tokens_store:
        return jsonify({"error": "Thiếu tokens để xoá. Bản ghi có thể được tạo trước khi lưu tokens."}), 400

    # Remove from stores
    app.tokens_store.pop(record_id, None)
    app.metadata_store.pop(record_id, None)
    save_metadata(app.metadata_store)
    save_tokens_store(app.tokens_store)

    # Rebuild DB from tokens
    try:
        app.db.server.init_db(APSI_PARAMS, max_label_length=MAX_LABEL_LENGTH)
        for rid, toks in app.tokens_store.items():
            app.db.add_record(rid, toks)
        app.db.save()
    except Exception as e:
        return jsonify({"error": f"Lỗi khi rebuild DB: {e}"}), 500

    return jsonify({"ok": True, "deleted": record_id})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)