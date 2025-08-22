# webhook_handlers.py
from __future__ import annotations
import json, logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel

from post_gemini import db
from triggers import auto_process_user, get_trigger_config
from celery_tasks import process_user_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])

class FirebaseWebhookData(BaseModel):
    uid: str
    event_type: str  # user_created, user_updated, prediction_updated ...
    timestamp: str
    data: Dict[str, Any] = {}

class GenericWebhook(BaseModel):
    event: str
    data: Dict[str, Any]
    timestamp: str

@router.post("/firebase/user-change")
async def handle_firebase_user_change(webhook_data: FirebaseWebhookData, background: BackgroundTasks):
    try:
        config = get_trigger_config()
        if not config.enabled:
            return {"ok": True, "message": "Trigger system disabled"}

        uid = webhook_data.uid
        evt = webhook_data.event_type
        doc = db.collection("users").document(uid).get()
        if not doc.exists:
            raise HTTPException(404, f"User {uid} not found")
        data = doc.to_dict()

        # 반드시 prediction.top5가 있어야 처리
        pred = (data.get("prediction") or {}).get("top5")
        if not pred:
            return {"ok": True, "message": "No prediction yet"}

        should = False
        trig = "webhook"
        if evt in ("user_updated", "prediction_updated", "recommendations_generated"):
            should = True
            trig = f"webhook_{evt}"

        if should:
            try:
                process_user_task.delay(uid, trig)
                method = "celery"
            except Exception:
                background.add_task(auto_process_user, uid, trig)
                method = "background"
            return {"ok": True, "message": f"Queued {uid}", "method": method}
        else:
            return {"ok": True, "message": f"Event {evt} ignored"}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, f"Webhook failed: {e}")

@router.post("/generic")
async def handle_generic_webhook(webhook: GenericWebhook, background: BackgroundTasks):
    # 기존 내용 유지
    ...

