import os
from dotenv import load_dotenv

load_dotenv()

# 보안
HOOK_SECRET = os.getenv("HOOK_SECRET", "change-me")

# Firestore
FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", "/app/serviceAccountKey.json")

# 파일/모델 경로
CSV_PATH = os.getenv("CSV_PATH", "/app/직업군별_평균벡터.csv")
LINEAR_LAYER_PATH = os.getenv("LINEAR_LAYER_PATH", "/app/linear_layer.pth")
SENTENCE_MODEL_PATH = os.getenv("SENTENCE_MODEL_PATH", "/app/kobert_base_model")  # 폴더 or 허깅페이스 모델명

# 하이퍼파라미터
TOP_K_PER_CRITERION = int(os.getenv("TOP_K_PER_CRITERION", "10"))
ALPHA_BLEND = float(os.getenv("ALPHA_BLEND", "0.5"))
DEVICE = os.getenv("DEVICE", "cpu")  # 'cpu' 권장 (EC2 GPU 없으면)
