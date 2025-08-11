from fastapi import FastAPI, HTTPException, Header
from services.firebase_client import get_db
from predictor import run_prediction
from config import HOOK_SECRET

app = FastAPI()

@app.get("/healthz")
def health():
    return {"ok": True}

# (디버그/직접 호출용) 기존 엔드포인트 유지
@app.post("/predict-user/{user_id}")
def predict_user(user_id: str):
    db = get_db()
    doc_ref = db.collection("users").document(user_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="user not found")
    user_data = doc.to_dict()

    results, weights_os = run_prediction(user_id, user_data)
    doc_ref.update({
        "recommendations": results,
        "weights": {
            "objective_weight": weights_os["w_obj"],
            "subjective_weight": weights_os["w_sub"],
        },
        "processed": True
    })
    return {"user_id": user_id, "recommendations": results}

# Cloud Functions가 호출하는 웹훅
@app.post("/firestore-hook")
def firestore_hook(payload: dict, authorization: str = Header(None)):
    # 간단한 인증
    if authorization != f"Bearer {HOOK_SECRET}":
        raise HTTPException(status_code=401, detail="unauthorized")
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    db = get_db()
    doc_ref = db.collection("users").document(user_id)
    snap = doc_ref.get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="user not found")
    user_data = snap.to_dict()

    # 이미 처리된 문서는 skip (idempotent)
    if user_data.get("processed") is True and user_data.get("recommendations"):
        return {"ok": True, "skipped": True}

    results, weights_os = run_prediction(user_id, user_data)
    doc_ref.update({
        "recommendations": results,
        "weights": {
            "objective_weight": weights_os["w_obj"],
            "subjective_weight": weights_os["w_sub"],
        },
        "processed": True
    })
    return {"ok": True, "user_id": user_id}
