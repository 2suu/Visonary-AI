# main.py
from __future__ import annotations

import os
import logging
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 로드맵/후처리 라우터
from post_gemini import router as post_router
from triggers import router as triggers_router, scheduler
from webhook_handlers import router as webhook_router

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

def create_app() -> FastAPI:
    app = FastAPI(
        title="Career Post-Processor API",
        version="1.0.0",
        description="Firestore 변경을 감지해 직업 인사이트/로드맵을 생성하고 저장합니다.",
    )

    # CORS (필요 시 도메인 추가)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(post_router)       # /post/*
    app.include_router(triggers_router)   # /triggers/*
    app.include_router(webhook_router)    # /webhooks/*

    @app.get("/")
    def root():
        return {"ok": True, "service": "career-post-processor"}

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/readyz")
    def readyz():
        # 간단한 레디니스 엔드포인트 (확장 가능)
        return {"ok": True}

    @app.on_event("startup")
    async def on_startup():
        log.info("Starting Career Post-Processor API")
        scheduler.start()
        log.info("Background scheduler started automatically")

        # 디버그: 등록된 라우트 나열 (LOG_ROUTES=1 일 때만)
        if os.getenv("LOG_ROUTES", "1") == "1":
            for r in app.routes:
                log.info("ROUTE %s %s", getattr(r, "methods", None), r.path)

    @app.on_event("shutdown")
    async def on_shutdown():
        log.info("Shutting down Career Post-Processor API")
        scheduler.stop()

    return app


app = create_app()

# uvicorn main:app 로 실행하는 환경을 고려하여 __main__ 블록은 선택사항
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8010")),
        reload=False,
    )
