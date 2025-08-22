from __future__ import annotations
import time, threading
from typing import Any, Dict
from config import get_settings
from services.firebase_client import stream_users, save_prediction
from predictor import recommend_jobs

_cfg = get_settings()
_inflight: set[str] = set()
_lock = threading.Lock()

def _need_run(doc: Dict[str, Any]) -> bool:
    """입력지문 해시가 다르면 실행. (prediction 저장으로 인한 자기변경 루프 방지)"""
    last_hash = ((doc.get("system") or {}).get("input_fingerprint")) or ""
    # 예측에 필요한 필드가 전혀 없다면 실행 안함
    if not any(k in doc for k in ["성격","흥미","나이","성별","최소급여","경력","경제상황 및 대인관계","감정기복","스트레스해소","관심분야","최종학력"]):
        return False
    # 항상 재계산 후 비교 (recommend_jobs 안에서 새 해시 생성)
    return True if not last_hash else True  # 일단 true, 콜백 내부에서 비교

def _handle_change(user_id: str, data: Dict[str, Any]):
    if not _need_run(data):
        return
    with _lock:
        if user_id in _inflight:
            return
        _inflight.add(user_id)

    try:
        result = recommend_jobs(data)
        last_hash = ((data.get("system") or {}).get("input_fingerprint")) or ""
        new_hash = result.get("input_fingerprint")
        # 해시가 동일하면 스킵
        if last_hash and new_hash and last_hash == new_hash:
            return
        # 저장
        save_prediction(user_id, {k:v for k,v in result.items() if k!="input_fingerprint"}, input_fingerprint=new_hash)
        print(f"[watch] predicted -> {user_id}: {result.get('top5')}")
    except Exception as e:
        print(f"[watch] error for {user_id}: {e}")
    finally:
        with _lock:
            _inflight.discard(user_id)

def main():
    print("[watch] starting firestore watcher...")
    stop = stream_users(_handle_change)  # on_snapshot 리스너 핸들 반환
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n[watch] stopping...")
        stop.unsubscribe()  # 안전 종료

if __name__ == "__main__":
    main()
