from __future__ import annotations
from typing import Any, Dict, Callable, Iterable
from google.cloud import firestore
from google.oauth2 import service_account
from datetime import datetime, timezone
from config import get_settings

_db: firestore.Client | None = None

def _init_db() -> firestore.Client:
    global _db
    if _db:
        return _db
    cfg = get_settings()
    creds = service_account.Credentials.from_service_account_file(cfg.FIREBASE_KEY_PATH)
    _db = firestore.Client(project=cfg.FIREBASE_PROJECT_ID, credentials=creds)
    return _db

# ------------------- CRUD -------------------

def get_user_doc(user_id: str):
    db = _init_db()
    return db.collection("users").document(user_id)  # ✅ DocumentReference 반환

def save_prediction(user_id: str,
                    result: Dict[str, Any],
                    input_fingerprint: Optional[str] = None) -> str:
    db = _init_db()
    cfg = get_settings()
    data: Dict[str, Any] = {
        "prediction": result,
        "prediction_meta": {
            "by": cfg.WATCHER_NAME,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    }
    if input_fingerprint:
        # 무한루프 방지용: 마지막으로 처리한 입력지문 해시 저장
        data.setdefault("system", {})  # merge=True 로 들어가면 중첩 병합
        data["system"] = { "input_fingerprint": input_fingerprint }

    db.collection(cfg.FIRESTORE_ROOT_COLLECTION).document(user_id).set(data, merge=True)


# ------------------- Watch -------------------

ChangeHandler = Callable[[str, Dict[str, Any]], None]

def stream_users(on_change: ChangeHandler):
    """users 루트 콜렉션의 문서 변경을 구독한다."""
    db = _init_db()
    cfg = get_settings()
    col_ref = db.collection(cfg.FIRESTORE_ROOT_COLLECTION)

    def _callback(col_snapshot, changes, read_time):
        for ch in changes:
            # ADDED, MODIFIED, REMOVED 중 ADDED/MODIFIED만 처리
            if ch.type.name not in ("ADDED", "MODIFIED"):
                continue
            doc = ch.document
            user_id = doc.id
            data = doc.to_dict() or {}
            try:
                on_change(user_id, data)
            except Exception as e:
                # 콜백 내부에서 로깅/처리
                print(f"[watch] error on {user_id}: {e}")

    # on_snapshot 은 백그라운드 스레드에서 호출됨
    return col_ref.on_snapshot(_callback)
