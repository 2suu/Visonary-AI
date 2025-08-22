from __future__ import annotations
import asyncio, logging, threading, time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from google.cloud.firestore_v1 import FieldFilter

from post_gemini import _process_user_sync, db

logger = logging.getLogger("triggers")
router = APIRouter(prefix="/triggers", tags=["automation"])

# ----------------- Models -----------------
class TriggerConfig(BaseModel):
    enabled: bool = True
    check_interval_minutes: int = 5
    auto_process_new_users: bool = True
    auto_process_updated_users: bool = True
    max_retries: int = 3
    retry_delay_minutes: int = 10

class UserTriggerStatus(BaseModel):
    uid: str
    status: str  # pending, processing, done, error
    trigger_type: str
    created_at: str
    updated_at: str
    retry_count: int = 0
    error_message: Optional[str] = None

# ----------------- Helpers -----------------
def get_trigger_config() -> TriggerConfig:
    try:
        doc = db.collection("system").document("trigger_config").get()
        if doc.exists:
            return TriggerConfig(**(doc.to_dict() or {}))
        cfg = TriggerConfig()
        db.collection("system").document("trigger_config").set(cfg.model_dump())
        return cfg
    except Exception as e:
        logger.error(f"get_trigger_config error: {e}")
        return TriggerConfig()

def save_trigger_status(status: UserTriggerStatus):
    db.collection("trigger_status").document(status.uid).set(status.model_dump())

def get_trigger_status(uid: str) -> Optional[UserTriggerStatus]:
    doc = db.collection("trigger_status").document(uid).get()
    if doc.exists:
        return UserTriggerStatus(**(doc.to_dict() or {}))
    return None

# ----------------- Query builders -----------------
def get_pending_users() -> List[str]:
    q = db.collection("users").where(
        filter=FieldFilter("prediction.top5", "!=", None)
    ).stream()

    uids: List[str] = []
    for doc in q:
        data = doc.to_dict() or {}
        pred = (data.get("prediction") or {}).get("top5")
        if not pred:
            continue
        sys = data.get("system") or {}
        if sys.get("post_fingerprint") == sys.get("input_fingerprint"):
            continue
        if data.get("post_status") in ("processing", "done"):
            continue
        uids.append(doc.id)
    return uids

def get_updated_users(minutes_ago: int = 60) -> List[str]:
    cutoff = datetime.utcnow() - timedelta(minutes=minutes_ago)
    q = db.collection("users").where(
        filter=FieldFilter("updatedAt", ">=", cutoff)
    ).stream()

    uids: List[str] = []
    for doc in q:
        data = doc.to_dict() or {}
        pred = (data.get("prediction") or {}).get("top5")
        if not pred:
            continue
        sys = data.get("system") or {}
        if sys.get("post_fingerprint") != sys.get("input_fingerprint"):
            uids.append(doc.id)
    return uids

def get_failed_users_for_retry(max_retries: int, delay_minutes: int) -> List[str]:
    cutoff = datetime.utcnow() - timedelta(minutes=delay_minutes)
    query = db.collection("trigger_status").where(
        filter=FieldFilter("status", "==", "error")
    ).stream()

    retry: List[str] = []
    for d in query:
        st = UserTriggerStatus(**(d.to_dict() or {}))
        if st.retry_count >= max_retries:
            continue
        # ISO8601 "Z" → RFC 변환
        last = datetime.fromisoformat(st.updated_at.replace("Z", "+00:00")).replace(tzinfo=None)
        if last > cutoff:
            continue
        retry.append(st.uid)
    return retry

# ----------------- Workers -----------------
async def auto_process_user(uid: str, trigger_type: str = "auto"):
    try:
        status = UserTriggerStatus(
            uid=uid,
            status="processing",
            trigger_type=trigger_type,
            created_at=datetime.utcnow().isoformat() + "Z",
            updated_at=datetime.utcnow().isoformat() + "Z",
        )
        save_trigger_status(status)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _process_user_sync, uid)

        status.status = "done"
        status.updated_at = datetime.utcnow().isoformat() + "Z"
        save_trigger_status(status)
    except Exception as e:
        logger.exception(f"auto_process_user error: {e}")
        st = get_trigger_status(uid) or UserTriggerStatus(
            uid=uid, status="error", trigger_type=trigger_type,
            created_at=datetime.utcnow().isoformat()+"Z",
            updated_at=datetime.utcnow().isoformat()+"Z"
        )
        st.status = "error"
        st.error_message = str(e)
        st.retry_count += 1
        st.updated_at = datetime.utcnow().isoformat() + "Z"
        save_trigger_status(st)

async def check_and_process_users():
    cfg = get_trigger_config()
    if not cfg.enabled:
        return
    tasks = []
    if cfg.auto_process_new_users:
        for uid in get_pending_users():
            tasks.append(auto_process_user(uid, "auto_new"))
    if cfg.auto_process_updated_users:
        for uid in get_updated_users(cfg.check_interval_minutes * 2):
            tasks.append(auto_process_user(uid, "auto_updated"))
    for uid in get_failed_users_for_retry(cfg.max_retries, cfg.retry_delay_minutes):
        tasks.append(auto_process_user(uid, "retry"))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

# ----------------- API -----------------
@router.get("/config")
def get_config():
    return get_trigger_config().model_dump()

@router.post("/config")
def update_config(config: TriggerConfig):
    db.collection("system").document("trigger_config").set(config.model_dump())
    return {"ok": True}

@router.post("/trigger/{uid}")
async def manual_trigger(uid: str, background: BackgroundTasks):
    background.add_task(auto_process_user, uid, "manual")
    return {"ok": True, "message": f"Manual trigger for {uid}"}

@router.get("/status/{uid}")
def get_user_status(uid: str):
    st = get_trigger_status(uid)
    if not st:
        raise HTTPException(404, "Trigger status not found")
    return st.model_dump()

@router.get("/status")
def get_all_status(limit: int = 50, status_filter: Optional[str] = None):
    q = db.collection("trigger_status").limit(limit)
    if status_filter:
        q = q.where(filter=FieldFilter("status", "==", status_filter))
    items = [{"uid": d.id, **(d.to_dict() or {})} for d in q.stream()]
    return {"count": len(items), "items": items}

# ----------------- Background scheduler -----------------
class BackgroundScheduler:
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _run(self):
        while self.running:
            try:
                cfg = get_trigger_config()
                if cfg.enabled:
                    asyncio.run(check_and_process_users())
                time.sleep(cfg.check_interval_minutes * 60)
            except Exception as e:
                logger.exception(e)
                time.sleep(60)

scheduler = BackgroundScheduler()
