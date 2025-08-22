# main.py
from __future__ import annotations
from typing import Dict, Any
from fastapi import FastAPI, Body

from predictor import recommend_jobs
from services.firebase_client import get_user_doc, save_prediction

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

# 1) 바디로 직접 받아 예측
@app.post("/recommend")
def recommend(user: Dict[str, Any] = Body(...)):
    result = recommend_jobs(user)
    input_fp = result.get("input_fingerprint")
    # 저장 (원하면 주석 해제)
    save_prediction(
        user_id=str(user.get("id", "test")),
        result={k: v for k, v in result.items() if k != "input_fingerprint"},
        input_fingerprint=input_fp,
    )
    return {
        "input_fingerprint": input_fp,
        "result": {k: v for k, v in result.items() if k != "input_fingerprint"},
    }

# 2) Firestore의 사용자 문서로 예측
@app.post("/predict-user/{user_id}")
def predict_user(user_id: str):
    # Firestore에서 사용자 데이터 로드
    doc_ref = get_user_doc(user_id)
    snap = doc_ref.get()
    if not snap.exists:
        return {"error": f"user '{user_id}' not found in Firestore"}
    user_payload = snap.to_dict() or {}

    # 예측
    result = recommend_jobs(user_payload)
    input_fp = result.get("input_fingerprint")

    # 저장 (insights/{user_id})
    save_prediction(
        user_id=user_id,  # ❗ 여기서 user가 아니라 user_id를 넘겨야 함
        result={k: v for k, v in result.items() if k != "input_fingerprint"},
        input_fingerprint=input_fp,
    )
    return {
        "user_id": user_id,
        "input_fingerprint": input_fp,
        "result": {k: v for k, v in result.items() if k != "input_fingerprint"},
    }
