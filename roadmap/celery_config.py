# celery_config.py
from __future__ import annotations
import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Redis 설정
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Celery 앱 생성
celery_app = Celery(
    "roadmap_processor",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["celery_tasks"]
)

# Celery 설정
celery_app.conf.update(
    # 타임존 설정
    timezone="Asia/Seoul",
    enable_utc=True,
    
    # 작업 설정
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # 결과 만료 시간 (1일)
    result_expires=3600 * 24,
    
    # 작업 라우팅
    task_routes={
        "celery_tasks.process_user_task": {"queue": "roadmap_queue"},
        "celery_tasks.check_and_process_users_task": {"queue": "scheduler_queue"},
    },
    
    # 주기적 작업 설정 (Celery Beat)
    beat_schedule={
        "check-users-every-5-minutes": {
            "task": "celery_tasks.check_and_process_users_task",
            "schedule": 300.0,  # 5분마다
        },
        "cleanup-old-statuses": {
            "task": "celery_tasks.cleanup_old_statuses_task",
            "schedule": 3600.0 * 6,  # 6시간마다
        },
    },
    
    # 워커 설정
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    
    # 재시도 설정
    task_default_retry_delay=60,
    task_max_retries=3,
)
