from typing import Optional
import firebase_admin
from firebase_admin import credentials, firestore
from config import FIREBASE_KEY_PATH

_db = None

def get_db():
    global _db
    if _db is None:
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred)
        _db = firestore.client()
    return _db
