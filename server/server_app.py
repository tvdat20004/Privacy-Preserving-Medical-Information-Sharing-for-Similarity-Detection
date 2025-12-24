# server/server_app.py
import os
import sys
import json
import hashlib
from pathlib import Path
from flask import Flask, request, jsonify, Response, render_template, redirect, session, url_for

# Add project root to Python path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import Services
from common.config import APSI_PARAMS, MAX_LABEL_LENGTH
from server.services.record_manager import MedicalRecordManager
from server.services.search_service import SearchService
import multiprocessing
from apsi.utils import set_thread_count

TEMPLATE_ROOT = _project_root / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_ROOT))
app.secret_key = os.environ.get("APP_SECRET_KEY", "dev-secret")

# --- OPTIMIZATION ---
# try:
#     cpu_count = multiprocessing.cpu_count()
#     set_thread_count(cpu_count)
#     print(f"--- APSI Thread Count set to {cpu_count} ---")
# except Exception as e:
#     print(f"Warning: Could not set APSI thread count: {e}")

# --- INITIALIZATION ---

print("--- Initializing Server Services ---")
record_manager = MedicalRecordManager(_project_root)
search_service = SearchService(record_manager)

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
    if username == 'admin' and hashlib.md5(password.encode()).hexdigest() == '827ccb0eea8a706c4c34a16891f84e7b': # 12345
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
    return Response(record_manager.handle_oprf(request.get_data()), mimetype='application/octet-stream')


import time

@app.route('/query', methods=['POST'])
def query():
    start_time = time.time()
    response = record_manager.handle_query(request.get_data())
    elapsed_time = time.time() - start_time
    print(f"[Query] Processed in {elapsed_time:.4f} seconds")
    return Response(response, mimetype='application/octet-stream')


@app.route('/metadata/<record_id>', methods=['GET'])
def get_metadata(record_id):
    meta = record_manager.get_metadata(record_id)
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

    try:
        matches = search_service.process_image_search(file)
        if not matches:
            return jsonify({"matches": [], "message": "Không tìm thấy bản ghi phù hợp"})
        return jsonify({"matches": matches})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/encode', methods=['POST'])
def api_encode():
    if 'image' not in request.files:
        return jsonify({"error": "Thiếu file ảnh (field name: image)"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "File ảnh không hợp lệ"}), 400

    try:
        tokens = search_service.encode_image(file)
        payload = {"tokens": tokens}
        return jsonify({"count": len(tokens), "enc": payload})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
            data = request.get_json(force=True, silent=True)
            tokens = data.get('tokens') if data else None
        except Exception:
            tokens = None

    if not tokens:
        return jsonify({"error": "Không tìm thấy tokens hợp lệ"}), 400

    print(f"[API] Search tokens request: {len(tokens)} tokens")
    if len(tokens) > 0:
        print(f"[API] First token type: {type(tokens[0])}")
        print(f"[API] First token value: {tokens[0]}")

    try:
        matches = search_service.process_token_search(tokens)
        if not matches:
            return jsonify({"matches": [], "message": "Không tìm thấy bản ghi phù hợp"})
        return jsonify({"matches": matches})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
    
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
    except ValueError:
        page = 1
        limit = 20

    all_records = record_manager.get_all_metadata()
    # Sort for consistent pagination
    all_records.sort(key=lambda x: x.get('record_id', ''))
    
    total_count = len(all_records)
    start = (page - 1) * limit
    end = start + limit
    
    records = all_records[start:end]
    
    return jsonify({
        "count": total_count,
        "records": records,
        "page": page,
        "limit": limit
    })


@app.route('/admin/upload', methods=['POST'])
def admin_upload():
    if not require_admin():
        return jsonify({"error": "Unauthorized"}), 401

    if 'image' not in request.files:
        return jsonify({"error": "Thiếu file ảnh (field name: image)"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "File ảnh không hợp lệ"}), 400

    diagnosis_str = request.form.get('diagnosis') or ''
    diagnosis_list = [d.strip() for d in diagnosis_str.split(',') if d.strip()]
    if not diagnosis_list:
        diagnosis_list = ["N/A"]

    age_raw = request.form.get('age') or ''
    name = request.form.get('name') or ''
    try:
        age = int(age_raw)
    except Exception:
        age = None

    # Save to temp file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or '.jpg') as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        record_id = record_manager.add_patient_record(temp_path, diagnosis_list, age, name)
        os.unlink(temp_path)
        return jsonify({"ok": True, "record_id": record_id})
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({"error": str(e)}), 500


@app.route('/admin/delete', methods=['POST'])
def admin_delete():
    if not require_admin():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    record_id = data.get('record_id')
    if not record_id:
        return jsonify({"error": "Thiếu record_id"}), 400

    try:
        success = record_manager.delete_patient_record(record_id)
        if success:
            return jsonify({"ok": True, "deleted": record_id})
        else:
            return jsonify({"error": "Record not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)