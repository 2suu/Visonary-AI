import math
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from services.model import get_models
from config import CSV_PATH, TOP_K_PER_CRITERION, ALPHA_BLEND, DEVICE

# CSV 메모리 로드
_df = None
_job_names = None
_df_features = None

def get_df():
    global _df, _job_names, _df_features
    if _df is None:
        _df = pd.read_csv(CSV_PATH)
        _job_names = _df["job"]
        _df_features = _df.drop(columns=["job"])
    return _df, _job_names, _df_features

# ===== 유틸 =====
def safe_get_value(field):
    if isinstance(field, dict):
        return field.get("value", "") if isinstance(field.get("value"), str) else field.get("value", 0)
    return field if isinstance(field, (int, float, str)) else 0

def safe_get_priority(field):
    if isinstance(field, dict):
        return float(field.get("priority", 1))
    return float(field) if isinstance(field, (int, float)) else 1.0

def normalize_priority(priorities):
    max_val = max(priorities) if max(priorities) > 0 else 1.0
    return [p / max_val for p in priorities]

def minmax(x):
    x = np.asarray(x, dtype=float)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

# ===== 점수 함수 =====
def score_age(user_age, job_age, sigma=5):
    diff = abs(user_age - job_age)
    return math.exp(- (diff ** 2) / (2 * sigma ** 2))

def score_salary(user_salary, job_salary):
    diff = abs(user_salary - job_salary)
    return 1 / (1 + diff / 500)

def score_trait(user_score, job_score):
    diff = abs(user_score - job_score)
    return 1 - (diff / 4)

def score_economic(user_value, job_value):
    diff = abs(user_value - job_value)
    return 1 - (diff / 4)

# ===== 예측 로직 =====
def predict_objective_components(user_data, df):
    p_age = safe_get_priority(user_data.get("연령"))
    p_salary = safe_get_priority(user_data.get("임금근로자 근로소득(연봉)"))
    p_econ = safe_get_priority(user_data.get("가정생활이나 사회생활 어려움"))
    p_person = safe_get_priority(user_data.get("성격"))

    p_age, p_salary, p_econ, p_person = normalize_priority([p_age, p_salary, p_econ, p_person])

    u_age = safe_get_value(user_data.get("연령"))
    u_salary = safe_get_value(user_data.get("임금근로자 근로소득(연봉)"))
    u_econ = safe_get_value(user_data.get("가정생활이나 사회생활 어려움"))

    trait_mapping = {
        "사회성": ["<성격> 사회성"],
        "성실성": ["<성격> 성취/노력", "<성격> 인내", "<성격> 책임성과 진취성", "<성격> 꼼꼼함", "<성격> 정직성"],
        "신경성": ["<성격> 스트레스감내성", "<성격> 자기통제"],
        "외향성": ["<성격> 리더십", "<성격> 적응성/융통성"],
        "친화성": ["<성격> 협조", "<성격> 타인에대한 배려", "<성격> 신뢰성"],
    }

    obj_age, obj_salary, obj_econ, obj_person = [], [], [], []
    for _, row in df.iterrows():
        obj_age.append(score_age(u_age, row["연령"]))
        obj_salary.append(score_salary(u_salary, row["임금근로자 근로소득(연봉)"]))
        obj_econ.append(score_economic(u_econ, row["가정생활이나 사회생활 어려움"]))
        trait_scores_all = []
        for big5_name, csv_cols in trait_mapping.items():
            user_trait_val = user_data["성격"].get(big5_name, 0)
            per_cols = []
            for col in csv_cols:
                if col in df.columns:
                    per_cols.append(score_trait(user_trait_val, row[col]))
            if per_cols:
                trait_scores_all.append(np.mean(per_cols))
        obj_person.append(np.mean(trait_scores_all) if trait_scores_all else 0.0)

    obj_age = np.array(obj_age)
    obj_salary = np.array(obj_salary)
    obj_econ = np.array(obj_econ)
    obj_person = np.array(obj_person)

    partials = [
        obj_age * p_age,
        obj_salary * p_salary,
        obj_econ * p_econ,
        obj_person * p_person,
    ]
    obj_sum = np.mean(partials, axis=0)

    return {
        "scores": {"age": obj_age, "salary": obj_salary, "economic": obj_econ, "personality": obj_person},
        "priorities": {"age": p_age, "salary": p_salary, "economic": p_econ, "personality": p_person},
        "sum_score": obj_sum,
    }

def predict_subjective_components(user_data, df_features):
    base_model, linear_layer = get_models()
    interest_text = safe_get_value(user_data.get("관심분야"))
    value_text = safe_get_value(user_data.get("가치관"))
    p_int = safe_get_priority(user_data.get("관심분야"))
    p_val = safe_get_priority(user_data.get("가치관"))
    p_int, p_val = normalize_priority([p_int, p_val])

    with torch.no_grad():
        emb_int = base_model.encode([interest_text], convert_to_tensor=True, device=DEVICE)
        emb_val = base_model.encode([value_text], convert_to_tensor=True, device=DEVICE)
        out_int = linear_layer(emb_int).cpu().numpy()
        out_val = linear_layer(emb_val).cpu().numpy()

    feats = df_features.values
    sub_int = cosine_similarity(out_int, feats)[0]
    sub_val = cosine_similarity(out_val, feats)[0]
    sub_sum = (sub_int * p_int + sub_val * p_val) / max((p_int + p_val), 1e-9)

    return {"scores": {"interest": sub_int, "value": sub_val}, "priorities": {"interest": p_int, "value": p_val}, "sum_score": sub_sum, "total_priority": (p_int + p_val)}

def hybrid_combine(obj_comp, sub_comp, job_names):
    p_obj_mean = np.mean(list(obj_comp["priorities"].values()))
    p_sub_sum = sub_comp["total_priority"]
    total_p = p_obj_mean + p_sub_sum if (p_obj_mean + p_sub_sum) > 0 else 1.0
    w_obj = p_obj_mean / total_p
    w_sub = p_sub_sum / total_p

    obj_sum_n = minmax(obj_comp["sum_score"])
    sub_sum_n = minmax(sub_comp["sum_score"])
    blended_sum = obj_sum_n * w_obj + sub_sum_n * w_sub

    crit_scores = {
        "age":      minmax(obj_comp["scores"]["age"]),
        "salary":   minmax(obj_comp["scores"]["salary"]),
        "economic": minmax(obj_comp["scores"]["economic"]),
        "personality": minmax(obj_comp["scores"]["personality"]),
        "interest": minmax(sub_comp["scores"]["interest"]),
        "value":    minmax(sub_comp["scores"]["value"])
    }
    crit_prios = {
        "age": obj_comp["priorities"]["age"],
        "salary": obj_comp["priorities"]["salary"],
        "economic": obj_comp["priorities"]["economic"],
        "personality": obj_comp["priorities"]["personality"],
        "interest": sub_comp["priorities"]["interest"],
        "value": sub_comp["priorities"]["value"]
    }

    p_list = np.array(list(crit_prios.values()), dtype=float)
    p_sum = p_list.sum() if p_list.sum() > 0 else 1.0
    crit_weights = {k: (v / p_sum) for k, v in crit_prios.items()}

    n = len(job_names)
    per_criterion_score = np.zeros(n, dtype=float)
    for name, arr in crit_scores.items():
        w = crit_weights[name]
        idx = np.argsort(arr)[::-1][:TOP_K_PER_CRITERION]
        ranks = np.arange(1, TOP_K_PER_CRITERION + 1, dtype=float)
        rank_bonus = (TOP_K_PER_CRITERION - ranks + 1) / TOP_K_PER_CRITERION
        per_criterion_score[idx] += rank_bonus * w

    per_criterion_score = minmax(per_criterion_score)
    final_score = ALPHA_BLEND * blended_sum + (1 - ALPHA_BLEND) * per_criterion_score
    return final_score, blended_sum, per_criterion_score, {"w_obj": w_obj, "w_sub": w_sub}

def run_prediction(user_id: str, user_data: dict):
    df, job_names, df_features = get_df()

    obj_comp = predict_objective_components(user_data, df)
    sub_comp = predict_subjective_components(user_data, df_features)
    final_score, sum_score_blended, percrit_score, weights_os = hybrid_combine(obj_comp, sub_comp, job_names)

    top_idx = np.argsort(final_score)[::-1][:5]
    jobs_series = job_names.reset_index(drop=True)
    obj_sum_norm = minmax(obj_comp["sum_score"])
    sub_sum_norm = minmax(sub_comp["sum_score"])

    results = []
    for i in top_idx:
        results.append({
            "job": str(jobs_series.iloc[i]),
            "final_score": float(final_score[i]),
            "blended_sum_score": float(sum_score_blended[i]),
            "per_criterion_boost": float(percrit_score[i]),
            "objective_sum_norm": float(obj_sum_norm[i]),
            "subjective_sum_norm": float(sub_sum_norm[i])
        })
    return results, weights_os
