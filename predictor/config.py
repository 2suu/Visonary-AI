from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict  # ✅ pydantic-settings 사용


class Settings(BaseSettings):
    # ---- Firestore / 파일 경로 ----
    FIREBASE_KEY_PATH: str = "serviceAccountKey.json"
    FIREBASE_PROJECT_ID: Optional[str] = None
    FIRESTORE_ROOT_COLLECTION: str = "users"

    # ---- 데이터/예측 ----
    CSV_PATH: str = "직업별_전체항목_평균_포함.csv"
    TOPK: int = 5

    # ---- watcher ----
    WATCH_FIELDS_HASH_KEY: str = "system.input_fingerprint"
    WATCHER_NAME: str = "predictor"
    WATCH_DEBOUNCE_MS: int = 400

    # ✅ pydantic v2 설정 (옛날 class Config 대체)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",              # 환경변수에 낯선 키가 있어도 무시
        env_nested_delimiter="__",   # 중첩 키 쓸 때 구분자
    )


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    # 경로는 절대경로로 정규화
    s.FIREBASE_KEY_PATH = str(Path(s.FIREBASE_KEY_PATH).resolve())
    s.CSV_PATH = str(Path(s.CSV_PATH).resolve())
    return s
