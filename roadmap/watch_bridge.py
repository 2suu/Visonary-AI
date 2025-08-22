# watch_bridge.py
from __future__ import annotations
import threading, time
from firebase_admin import credentials, firestore, initialize_app, _apps
from post_gemini import _process_user_sync, FIREBASE_KEY_PATH

if not _apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    initialize_app(cred)
db = firestore.client()

_inflight = set()
_lock = threading.Lock()

def _handle_change(doc_snapshot, changes, read_time):
    for ch in changes:
        if ch.type.name not in ("ADDED", "MODIFIED"): 
            continue
        doc = ch.document
        uid = doc.id
        data = doc.to_dict() or {}
        pred = (data.get("prediction") or {}).get("top5")
        if not pred:
            continue
        sys = data.get("system") or {}
        if sys.get("post_fingerprint") == sys.get("input_fingerprint"):
            continue
        with _lock:
            if uid in _inflight: 
                continue
            _inflight.add(uid)
        try:
            _process_user_sync(uid)
        finally:
            with _lock:
                _inflight.discard(uid)

def main():
    print("[bridge] watching users for prediction → roadmap")
    stop = db.collection("users").on_snapshot(_handle_change)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        stop.unsubscribe()

if __name__ == "__main__":
    main()
