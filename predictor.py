# predictor.py
from __future__ import annotations

import json, hashlib, re, unicodedata
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import get_settings
from services.text_embedder import encode_texts


# ================== 로드/설정 ==================
_cfg = get_settings()
_df = pd.read_csv(_cfg.CSV_PATH)
if "job" not in _df.columns:
    raise ValueError("CSV에 'job' 컬럼이 없습니다.")
JOBS = _df["job"].tolist()
TOPK = int(getattr(_cfg, "TOPK", 5))

# 직업 임베딩(다양화용, 있으면 사용)
_JOB_VEC = None
for path in [getattr(_cfg, "JOB_VEC_PATH", None), "/mnt/data/직업군별_평균벡터.csv"]:
    try:
        if path:
            jv = pd.read_csv(path)
            if "job" in jv.columns:
                _JOB_VEC = jv
                break
    except Exception:
        pass


# ================== 매핑 ==================
TRAIT_MAPPING: Dict[str, List[str]] = {
    "개방성": ["<성격> 혁신", "<성격> 분석적 사고", "<성격> 독립성"],
    "성실성": ["<성격> 성취/노력", "<성격> 인내", "<성격> 책임성과 진취성", "<성격> 꼼꼼함", "<성격> 정직성"],
    "신경성": ["<성격> 스트레스감내성", "<성격> 자기통제"],
    "외향성": ["<성격> 리더십", "<성격> 사회성", "<성격> 적응성/융통성"],
    "친화성": ["<성격> 협조", "<성격> 타인에대한 배려", "<성격> 신뢰성"],
}
INTEREST_MAPPING: Dict[str, List[str]] = {
    "관습형": ["<지식> 사무_중요도", "<지식> 경영 및 행정_중요도", "<지식> 통신_중요도", "<지식> 법_중요도"],
    "기업형": ["<지식> 영업과 마케팅_중요도", "<지식> 경영 및 행정_중요도", "<지식> 법_중요도", "<지식> 고객서비스_중요도"],
    "사회형": ["<지식> 교육및훈련_중요도", "<지식> 상담_중요도", "<지식> 심리_중요도", "<지식> 사회와인류_중요도"],
    "실재형": ["<지식> 공학과 기술_중요도", "<지식> 기계_중요도", "<지식> 건축및설계_중요도", "<지식> 안전과보안_중요도", "<지식> 운송_중요도"],
    "예술형": ["<지식> 디자인_중요도", "<지식> 예술_중요도", "<지식> 의사소통과미디어_중요도"],
}
BASIC_MAPPING: Dict[str, str] = {"나이": "연령", "성별": "성별", "최종학력": "최종학력"}
SALARY_COLUMNS = ["임금근로자 근로소득(연봉)", "임금근로자 초임임금(연봉)", "비임금근로자 순수입(연봉)"]
ECON_SOCIAL_MAPPING = {
    "대인관계": ["<지식> 고객서비스_중요도", "<지식> 상담_중요도", "직무만족_동료", "직무만족_상사"],
    "관계 스타일": ["<성격> 리더십", "<성격> 사회성", "<성격> 적응성/융통성"],
    "생활비": ["일과 생활 균형_업무걱정", "일과 생활 균형_일 하는데 시간 부족", "근무조건", "직업 안정성"],
    "저축": ["일과 생활 균형_가족과 함께 보낼 수 있는 시간 부족", "일과 생활 균형_집안일을 못함", "근무조건", "직업 안정성"],
}
MOOD_STRESS_MAPPING = {
    "감정기복": [
        "정신건강_즐겁고 기분이 좋았다",
        "정신건강_마음이 차분하고 편안했다",
        "정신건강_활발하고 활기찼다",
        "정신건강_아침에 피로가 다 가셔서 상쾌하게 일어났다",
        "정신건강_일상생활은 흥미로운 것들로 가득 차있었다",
    ],
    "스트레스해소": ["<성격> 스트레스감내성"],
}
CAREER_COLUMNS = ["요구자격", "관련된 업무경험", "교육수준"]


# ================== 하이퍼파라미터 ==================
PRIORITY_TAU = 1.0          # 섹션 softmax 온도
SIM_GAMMA = 1.2             # 수치 유사도 샤프닝(>1이면 상위 강조)
IQR_SCALE = 1.5             # |x-t| / (IQR*scale)
INTEREST_TEXT_SIM_TH = 0.40 # (미사용) 단일 매핑 임계 — 하위호환용
SALARY_PENALTY_STRENGTH = 0.70   # 0~1 (클수록 급여 미달 패널티 큼)
SALARY_PENALTY_FLOOR    = 0.30   # 패널티 하한(아무리 부족해도 최소 이 비율은 남김)
MMR_LAMBDA = 0.65           # 다양화 강도

# 관심(주관식) 하이브리드 매핑 하이퍼파라미터
INTEREST_CAND_PREFIXES = ["<지식> ", "<능력> ", "<작업활동> ", "<일의 내용> "]
TECH_KEYWORDS = [
    "ai","인공지능","머신러닝","딥러닝","ml","dl","데이터","빅데이터","데이터분석","데이터 사이언스",
    "통계","수학","알고리즘","소프트웨어","소프트웨어개발","개발","프로그래밍","코딩","python","자바",
    "c++","자바스크립트","javascript","typescript","프론트엔드","frontend","백엔드","backend","웹","웹개발",
    "서비스개발","클라우드","aws","gcp","azure","mle","mlops","데브옵스","devops","컴퓨터","컴퓨터공학",
    "전자공학","시각화","시스템","서버","네트워크","보안","nlp","자연어처리","컴퓨터비전","cv","추천시스템",
    "react","fastapi","django","spring","sql","database","db"
]
INTEREST_ALPHA = 0.7         # 임베딩 비중
INTEREST_TOPN = 3            # 가중합에 포함할 상위 컬럼 수
INTEREST_KEEP_TH = 0.45      # 하이브리드 점수 임계(낮으면 버림)


# ================== 유틸 ==================
def _num_value(node: Any, default: float = 0.0) -> float:
    try:
        if isinstance(node, dict):
            v = node.get("value", node.get("val", default))
        else:
            v = node
        return float(default if v is None else v)
    except Exception:
        return float(default)

def _priority_value(node: Any, default: int = 4) -> int:
    try:
        if isinstance(node, dict):
            p = node.get("priority", default)
        else:
            p = default
        p = int(p)
        return max(1, min(6, p))
    except Exception:
        return default

def _softmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return x
    z = x - np.max(x)
    e = np.exp(z)
    s = e.sum()
    return e / (s if s > 0 else 1.0)

def _section_weights(prios: Dict[str, int]) -> Dict[str, float]:
    names = list(prios.keys())
    imp = np.array([(7 - int(prios[n])) / PRIORITY_TAU for n in names], dtype=float)
    w = _softmax(imp)
    return {n: float(w[i]) for i, n in enumerate(names)}

def _combine_columns(cols: List[str]) -> List[str]:
    return [c for c in cols if c in _df.columns]

def _robust_sim(series: pd.Series, target: float) -> np.ndarray:
    """IQR 기반 선형 유사도: sim = max(0, 1 - |x-t|/(IQR*scale))^gamma"""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    q1, q3 = np.nanpercentile(s, 25), np.nanpercentile(s, 75)
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 1e-9:
        std = float(np.nanstd(s))
        iqr = std if (np.isfinite(std) and std > 0) else 1.0
    d = np.abs(s - float(target)) / (iqr * IQR_SCALE)
    sim = np.maximum(0.0, 1.0 - d)
    sim = np.power(sim, SIM_GAMMA)
    sim = np.where(np.isnan(sim), 0.0, sim)
    return sim

def _max_sim_across(cols: List[str], target: float) -> np.ndarray:
    valid = _combine_columns(cols)
    if not valid: return np.zeros(len(_df))
    sims = np.vstack([_robust_sim(_df[c], target) for c in valid])
    return np.max(sims, axis=0)

def _percentile(arr: np.ndarray, power: float = 0.75) -> np.ndarray:
    """0~1 백분위(상위 강조 power<1)."""
    if arr.size == 0:
        return arr
    pct = pd.Series(arr).rank(pct=True, method="average").to_numpy()
    pct = np.power(pct, power)
    return pct

def _fingerprint_inputs(user: Dict[str, Any]) -> str:
    keys = ["감정기복","경력","경제상황 및 대인관계","관심분야","나이","성격","성별","스트레스해소","최소급여","최종학력","흥미"]
    pruned = {k: user.get(k) for k in keys if k in user}
    s = json.dumps(pruned, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ================== 관심(주관식) 하이브리드 매핑 ==================
SYNONYM_EXPANSION = {
    r"\b프론트\s*엔드\b": ["프론트엔드","frontend","웹","웹개발","자바스크립트","javascript","react","ui"],
    r"\b백\s*엔드\b": ["백엔드","backend","서버","데이터베이스","db","api","spring","django","fastapi"],
    r"\b빅\s*데이터\b": ["빅데이터","data","데이터","데이터분석","spark","hadoop","etl","warehouse","pipeline"],
    r"\bai\b": ["ai","인공지능","머신러닝","딥러닝","ml","dl","모델","학습","추론","mle","mlops"],
    r"\bnlp\b": ["nlp","자연어처리","텍스트","언어모델","llm","KoBERT","KoSimCSE"],
}

def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return s.lower().strip()

def _tokenize_interest(text: str) -> List[str]:
    t = _normalize(text)
    # 콤마/슬래시/개행/연속공백 기준 분리
    parts = re.split(r"[,\n/]|[\s]{2,}", t)
    parts = [p.strip() for p in parts if p.strip()]
    base = " ".join(parts) if parts else t
    # 동의어 확장
    expanded: List[str] = []
    for pat, repls in SYNONYM_EXPANSION.items():
        if re.search(pat, base):
            expanded.extend(repls)
    # 알파뉴메릭/한글 토큰
    tokens = re.findall(r"[a-zA-Z0-9+#]+|[가-힣]+", base)
    # 중복 제거(순서 유지)
    dedup = list(dict.fromkeys(parts + tokens + expanded))
    return dedup

def _candidate_interest_columns() -> List[str]:
    cols: List[str] = []
    for c in _df.columns:
        if any(c.startswith(p) for p in INTEREST_CAND_PREFIXES):
            cols.append(c)
    return cols

def _keyword_overlap_score(colname: str, tokens: List[str]) -> float:
    # 접두/접미 정리 후 비교
    name = colname
    for p in INTEREST_CAND_PREFIXES:
        name = name.replace(p, "")
    name = name.replace("_중요도", "")
    name_norm = _normalize(name)

    score = 0.0
    for tk in tokens:
        tk_norm = _normalize(tk)
        if tk_norm and tk_norm in name_norm:
            score += 1.0
    if any(k in name_norm for k in TECH_KEYWORDS):
        score += 0.5
    return score

def _best_interest_columns_from_text(text: str, topn: int = INTEREST_TOPN) -> List[Tuple[str, float]]:
    """
    입력 텍스트를 기반으로 관심 관련 후보 컬럼들 중 상위 N개를 반환.
    반환값: [(column_name, hybrid_score)]
    """
    cands = _candidate_interest_columns()
    if not text or not cands:
        return []

    tokens = _tokenize_interest(text)

    # 임베딩: '핵심 문장' + 상위 몇 개 토큰 → 후보 라벨과 비교
    q_texts = [" ".join(tokens)] + tokens[:5] if tokens else [text]
    q_emb = encode_texts(q_texts)

    label_texts = []
    for c in cands:
        name = c
        for p in INTEREST_CAND_PREFIXES:
            name = name.replace(p, "")
        name = name.replace("_중요도", "")
        label_texts.append(name)
    k_emb = encode_texts(label_texts)

    emb_sims = cosine_similarity(q_emb, k_emb).max(axis=0)  # (K,)

    kw_scores = np.array([_keyword_overlap_score(c, tokens) for c in cands], dtype=float)
    if kw_scores.max() > 0:
        kw_scores = kw_scores / kw_scores.max()

    hybrid = INTEREST_ALPHA * emb_sims + (1.0 - INTEREST_ALPHA) * kw_scores
    order = np.argsort(-hybrid)[:max(1, topn)]
    return [(cands[i], float(hybrid[i])) for i in order]


# ================== 섹션 스코어(0~1) ==================
def _score_traits(user: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
    trait = user.get("성격")
    if not isinstance(trait, dict):
        return np.zeros(len(_df)), {}
    sims = np.zeros(len(_df)); used = []
    for k, cols in TRAIT_MAPPING.items():
        v = _num_value(trait.get(k), 0.0)
        cols_in = _combine_columns(cols)
        if not cols_in: continue
        used += cols_in
        sims += _max_sim_across(cols_in, v)
    if not used: return np.zeros(len(_df)), {}
    sims = sims / len(set(used))
    return _percentile(sims), {"columns": used}

def _score_interests(user: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
    intr = user.get("흥미")
    if not isinstance(intr, dict):
        return np.zeros(len(_df)), {}
    sims = np.zeros(len(_df)); used = []
    for k, cols in INTEREST_MAPPING.items():
        v = _num_value(intr.get(k), 0.0)
        cols_in = _combine_columns(cols)
        if not cols_in: continue
        used += cols_in
        sims += _max_sim_across(cols_in, v)
    if not used: return np.zeros(len(_df)), {}
    sims = sims / len(set(used))
    return _percentile(sims), {"columns": used}

def _score_basics(user: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
    sims = np.zeros(len(_df)); used = []
    for field, col in BASIC_MAPPING.items():
        if field in user and col in _df.columns:
            val = _num_value(user[field])
            sims += _robust_sim(_df[col], val)
            used.append(col)
    if not used: return np.zeros(len(_df)), {}
    sims = sims / len(used)
    return _percentile(sims), {"columns": used}

def _score_min_salary(user: Dict[str, Any]) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    섹션 점수(0~1) + 패널티 벡터 반환 (게이팅 제거)
    패널티는 최종 점수에 곱해지는 [SALARY_PENALTY_FLOOR, 1] 값
    """
    n = len(_df)
    if "최소급여" not in user:
        return np.zeros(n), {}, np.ones(n, dtype=float)

    min_salary = _num_value(user.get("최소급여"), 0.0)
    valid = [c for c in SALARY_COLUMNS if c in _df.columns]
    if not valid or min_salary <= 0:
        return np.zeros(n), {"columns": valid}, np.ones(n, dtype=float)

    arrs = np.vstack([pd.to_numeric(_df[c], errors="coerce").fillna(0).to_numpy() for c in valid])
    job_max = np.max(arrs, axis=0)

    # 여유비율(0~1): 1이면 충분, 0이면 전혀 미달
    ratio = np.clip(job_max / max(1.0, min_salary), 0.0, 1.0)

    # (노출용) 섹션 점수
    sims = _percentile(ratio)

    # (랭킹/최종 반영) 패널티
    penalty = 1.0 - SALARY_PENALTY_STRENGTH * (1.0 - ratio)
    penalty = np.clip(penalty, SALARY_PENALTY_FLOOR, 1.0)

    return sims, {"columns": valid, "min_salary": float(min_salary)}, penalty

def _score_econ_social(user: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
    econ = user.get("경제상황 및 대인관계")
    if not isinstance(econ, dict):
        return np.zeros(len(_df)), {}
    sims = np.zeros(len(_df)); used = []
    for sub_k, cols in ECON_SOCIAL_MAPPING.items():
        v = _num_value(econ.get(sub_k), 0.0)
        cols_in = _combine_columns(cols)
        if not cols_in: continue
        used += cols_in
        sims += _max_sim_across(cols_in, v)
    if not used: return np.zeros(len(_df)), {}
    sims = sims / len(set(used))
    return _percentile(sims), {"columns": used}

def _score_mood_stress(user: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
    sims = np.zeros(len(_df)); used = []
    if "감정기복" in user:
        target = 5.0 - _num_value(user["감정기복"])
        cols = _combine_columns(MOOD_STRESS_MAPPING["감정기복"])
        if cols:
            used += cols
            sims += _max_sim_across(cols, target)
    if "스트레스해소" in user:
        val = _num_value(user["스트레스해소"])
        cols = _combine_columns(MOOD_STRESS_MAPPING["스트레스해소"])
        if cols:
            used += cols
            sims += _max_sim_across(cols, val)
    if not used: return np.zeros(len(_df)), {}
    sims = sims / len(set(used))
    return _percentile(sims), {"columns": used}

def _score_career(user: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
    car = user.get("경력")
    if not isinstance(car, dict):
        return np.zeros(len(_df)), {}
    used = _combine_columns(CAREER_COLUMNS)
    if not used: return np.zeros(len(_df)), {}
    years = _num_value(car.get("기간"), 0.0)
    has = 1.0 if bool(car.get("value", False)) else 0.0
    target = min(5.0, years / 2.0) * has
    sims = _max_sim_across(used, target)
    return _percentile(sims), {"columns": used}

def _score_free_interest(user: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
    """
    주관식 '관심분야'를 하이브리드(임베딩+키워드)로
    <지식>/<능력>/<작업활동>/… 계열 상위 다중 컬럼에 매핑하여 가중합 점수를 산출
    """
    node = user.get("관심분야")
    if node is None:
        return np.zeros(len(_df)), {}
    text = str(node.get("value", node) if isinstance(node, dict) else node).strip()
    if not text:
        return np.zeros(len(_df)), {}

    # 후보 컬럼 상위 N개 추출
    top_cols = _best_interest_columns_from_text(text, topn=INTEREST_TOPN)
    if not top_cols:
        return np.zeros(len(_df)), {"mapped": None, "sim": 0.0}

    # 임계치 이하 제외
    sims_keep = [(c, s) for c, s in top_cols if s >= INTEREST_KEEP_TH]
    if not sims_keep:
        return np.zeros(len(_df)), {"mapped": None, "sim": float(max(s for _, s in top_cols))}

    # 가중합(컬럼별 하이브리드 점수를 weight로 사용)
    total_w = sum(s for _, s in sims_keep)
    sims = np.zeros(len(_df))
    used: List[str] = []
    for col, w in sims_keep:
        if col not in _df.columns:
            continue
        used.append(col)
        sims += (w / total_w) * _robust_sim(_df[col], 5.0)

    return _percentile(sims), {
        "columns": used,
        "mapped": [c for c, _ in sims_keep],
        "hybrid_scores": {c: float(s) for c, s in sims_keep},
    }


# ================== 다양화(MMR) ==================
def _job_feature_matrix() -> Tuple[np.ndarray, List[str]]:
    if _JOB_VEC is not None:
        jobs = _JOB_VEC["job"].tolist()
        cols = [c for c in _JOB_VEC.columns if c != "job"]
        X = _JOB_VEC[cols].to_numpy(dtype=float)
        order = [jobs.index(j) if j in jobs else -1 for j in JOBS]
        mask = np.array(order) >= 0
        X2 = np.zeros((len(JOBS), X.shape[1]))
        X2[mask] = X[np.array(order)[mask]]
        return X2, JOBS
    # fallback: CSV 수치컬럼 표준화
    num_cols = _df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return np.eye(len(JOBS)), JOBS
    X = _df[num_cols].to_numpy(dtype=float)
    cmin = np.nanmin(X, axis=0); cmax = np.nanmax(X, axis=0)
    denom = np.where(cmax - cmin > 1e-9, cmax - cmin, 1.0)
    X = (np.nan_to_num(X, nan=cmin) - cmin) / denom
    return X, JOBS

def _mmr_select(scores: np.ndarray, k: int, lam: float = MMR_LAMBDA) -> List[int]:
    n = scores.shape[0]
    X, _ = _job_feature_matrix()
    if n != X.shape[0]:
        X = np.pad(X, ((0, n - X.shape[0]), (0, 0)), mode="constant")[:n]
    sims = cosine_similarity(X)  # n x n
    selected: List[int] = []
    cand = set(range(n))
    while len(selected) < min(k, n) and cand:
        best_i, best_val = None, -1e9
        for i in cand:
            div = 0.0 if not selected else float(np.max(sims[i, selected]))
            val = lam * float(scores[i]) - (1.0 - lam) * div
            if val > best_val:
                best_val, best_i = val, i
        selected.append(best_i)  # type: ignore[arg-type]
        cand.remove(best_i)      # type: ignore[arg-type]
    return selected


# ================== 메인 ==================
def recommend_jobs(user: Dict[str, Any]) -> Dict[str, Any]:
    # 1) 섹션 가중치
    section_prios = {
        "성격": _priority_value(user.get("성격"), 4),
        "흥미": _priority_value(user.get("흥미"), 4),
        "기본": max(
            _priority_value(user.get("나이"), 4),
            _priority_value(user.get("성별"), 4),
            _priority_value(user.get("최종학력"), 4),
        ),
        "최소급여": _priority_value(user.get("최소급여"), 4),
        "경제/대인": _priority_value(user.get("경제상황 및 대인관계"), 4),
        "정서/스트레스": 4,
        "경력": _priority_value(user.get("경력"), 4),
        "관심(주관식)": _priority_value(user.get("관심분야"), 4),
    }
    W_all = _section_weights(section_prios)

    # 2) 섹션 점수 계산
    sims_map: Dict[str, Tuple[np.ndarray, Dict]] = {
        "성격": _score_traits(user),
        "흥미": _score_interests(user),
        "기본": _score_basics(user),
        "경제/대인": _score_econ_social(user),
        "정서/스트레스": _score_mood_stress(user),
        "경력": _score_career(user),
        "관심(주관식)": _score_free_interest(user),
    }
    salary_sims, salary_meta, salary_penalty = _score_min_salary(user)
    sims_map["최소급여"] = (salary_sims, salary_meta)

    # 활성 섹션만 가중치 재정규화
    active_names = [n for n, (s, _m) in sims_map.items() if np.any(s > 0)]
    if not active_names:
        active_names = list(sims_map.keys())
    active_w = {n: W_all[n] for n in active_names}
    ssum = sum(active_w.values()) or 1.0
    W = {n: active_w[n] / ssum for n in active_names}

    # 3) 가중 합산
    agg = np.zeros(len(JOBS))
    breakdown = {"section_weights": W, "sections": {}}
    for name in active_names:
        sims, meta = sims_map[name]
        agg += W[name] * sims
        m = dict(meta)
        m["weight"] = W[name]
        breakdown["sections"][name] = m

    # 4) 랭킹/노출 점수 (게이팅 X, 패널티 O)
    agg_penalized = agg * salary_penalty
    ranking_scores = agg_penalized
    display_scores = np.clip(agg_penalized, 0.0, 1.0)

    # 5) 최종 순위 + 다양화
    order = np.argsort(-ranking_scores)
    if len(JOBS) > 10:
        picked = _mmr_select(ranking_scores, k=TOPK)
    else:
        picked = order[:TOPK].tolist()

    top_jobs = [JOBS[i] for i in picked]
    all_scores = {JOBS[i]: float(display_scores[i]) for i in order}

    # 6) (선택) 패널티 요약 / excluded_by_salary 제거
    worst_idx = np.argsort(salary_penalty)[:10].tolist()
    breakdown["salary_penalty_summary"] = {
        "min_penalty": float(np.min(salary_penalty)),
        "mean_penalty": float(np.mean(salary_penalty)),
        "most_penalized": [{"job": JOBS[i], "penalty": float(salary_penalty[i])} for i in worst_idx],
    }

    return {
        "top5": top_jobs,
        "scores": all_scores,
        "breakdown": breakdown,
        "input_fingerprint": _fingerprint_inputs(user),
    }
