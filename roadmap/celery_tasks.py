# celery_tasks.py
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import List

from celery import current_task
from celery_config import celery_app
from post_gemini import _process_user_sync, db
from triggers import (
    get_trigger_config, 
    save_trigger_status, 
    get_trigger_status, 
    get_pending_users, 
    get_updated_users,
    get_failed_users_for_retry,
    UserTriggerStatus
)

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3)
def process_user_task(self, uid: str, trigger_type: str = "celery"):
    """Celery를 통한 사용자 처리 작업"""
    try:
        # 트리거 상태 생성/업데이트
        status = UserTriggerStatus(
            uid=uid,
            status="processing",
            trigger_type=trigger_type,
            created_at=datetime.utcnow().isoformat() + "Z",
            updated_at=datetime.utcnow().isoformat() + "Z"
        )
        save_trigger_status(status)
        
        logger.info(f"[Celery] Starting processing for user {uid}")
        
        # 실제 처리
        _process_user_sync(uid)
        
        # 성공 상태 업데이트
        status.status = "done"
        status.updated_at = datetime.utcnow().isoformat() + "Z"
        save_trigger_status(status)
        
        logger.info(f"[Celery] Completed processing for user {uid}")
        
        return {"uid": uid, "status": "completed"}
        
    except Exception as e:
        logger.error(f"[Celery] Error processing user {uid}: {e}")
        
        # 에러 상태 업데이트
        status = get_trigger_status(uid)
        if status:
            status.status = "error"
            status.error_message = str(e)
            status.retry_count = self.request.retries
            status.updated_at = datetime.utcnow().isoformat() + "Z"
            save_trigger_status(status)
        
        # Celery 재시도
        raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))

@celery_app.task
def check_and_process_users_task():
    """주기적으로 사용자들을 확인하고 처리하는 작업"""
    try:
        config = get_trigger_config()
        
        if not config.enabled:
            logger.info("[Celery] Trigger system disabled, skipping check")
            return {"message": "disabled"}
        
        logger.info("[Celery] Checking for users to process...")
        
        processed_users = []
        
        # 새로운 사용자 처리
        if config.auto_process_new_users:
            pending_users = get_pending_users()
            for uid in pending_users:
                process_user_task.delay(uid, "auto_new")
                processed_users.append({"uid": uid, "type": "new"})
        
        # 업데이트된 사용자 처리
        if config.auto_process_updated_users:
            updated_users = get_updated_users(config.check_interval_minutes * 2)
            for uid in updated_users:
                process_user_task.delay(uid, "auto_updated")
                processed_users.append({"uid": uid, "type": "updated"})
        
        # 실패한 작업 재시도
        retry_users = get_failed_users_for_retry(config.max_retries, config.retry_delay_minutes)
        for uid in retry_users:
            process_user_task.delay(uid, "retry")
            processed_users.append({"uid": uid, "type": "retry"})
        
        logger.info(f"[Celery] Queued {len(processed_users)} users for processing")
        
        return {
            "processed_count": len(processed_users),
            "users": processed_users,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"[Celery] Error in check_and_process_users_task: {e}")
        raise

@celery_app.task
def cleanup_old_statuses_task():
    """오래된 트리거 상태 정리"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=7)  # 7일 이전
        
        query = db.collection("trigger_status").where(
            "updated_at", "<", cutoff_date.isoformat() + "Z"
        ).stream()
        
        deleted_count = 0
        for doc in query:
            doc.reference.delete()
            deleted_count += 1
        
        logger.info(f"[Celery] Cleaned up {deleted_count} old trigger statuses")
        
        return {"deleted_count": deleted_count}
        
    except Exception as e:
        logger.error(f"[Celery] Error in cleanup_old_statuses_task: {e}")
        raise

@celery_app.task
def batch_process_users_task(uids: List[str], trigger_type: str = "batch"):
    """여러 사용자를 배치로 처리"""
    results = []
    
    for uid in uids:
        try:
            process_user_task.delay(uid, trigger_type)
            results.append({"uid": uid, "status": "queued"})
        except Exception as e:
            logger.error(f"[Celery] Error queuing user {uid}: {e}")
            results.append({"uid": uid, "status": "failed", "error": str(e)})
    
    return {
        "total": len(uids),
        "results": results,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
