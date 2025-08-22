# post_gemini.py
from __future__ import annotations

import os
import re
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

import google.generativeai as genai

from firebase_admin import credentials, firestore, initialize_app, _apps

# ============================================================================
# Environment / SDK bootstrap
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", str(BASE_DIR / "serviceAccountKey.json"))

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

genai.configure(api_key=GEMINI_API_KEY)

if not _apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    initialize_app(cred)
db = firestore.client()

router = APIRouter(prefix="/post", tags=["postprocess"])

# ============================================================================
# Pydantic Schemas (Gemini 결과 스키마)
# ============================================================================
class RoadmapStep(BaseModel):
    title: str
    tasks: List[str]
    resources: List[str] = []
    milestone: Optional[str] = None
    expected_duration_weeks: Optional[int] = None


class Roadmap(BaseModel):
    overview: str
    total_estimate_months: int
    phases: List[RoadmapStep]


class Analysis(BaseModel):
    big5_alignment: Dict[str, str] = Field(default_factory=dict)
    interests_alignment: List[str] = Field(default_factory=list)
    values_alignment: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    action_advice: List[str] = Field(default_factory=list)


class JobInfo(BaseModel):
    summary: str
    key_tasks: List[str]
    core_skills: List[str]
    typical_tools: List[str] = []
    average_salary_krw: Optional[str] = None
    growth_outlook: Optional[str] = None
    related_jobs: List[str] = []


class JobInsight(BaseModel):
    analysis: Analysis
    job_info: JobInfo
    roadmap: Roadmap
    model: str = GEMINI_MODEL
    updatedAt: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")

# --------------------- UI Summary (프론트 전용 저장 포맷) ---------------------
class UIValueBar(BaseModel):
    label: str
    percent: int


class UIRadar(BaseModel):
    axes: List[UIValueBar]


class UICareerCard(BaseModel):
    job: str
    subtitle: str
    tags: List[str] = []
    avg_salary_text: Optional[str] = None


class UIRoadmapBlock(BaseModel):
    title: str
    items: List[Dict[str, Any]]  # {"text": str, "done": bool}
    progress: int = 0


class UISummary(BaseModel):
    big5: Dict[str, Any]
    interests: Dict[str, Any]
    affect_values: Dict[str, Any]
    overall: Dict[str, Any]
    career_options: List[UICareerCard]
    roadmaps: Dict[str, List[UIRoadmapBlock]]

# ============================================================================
# Firestore helpers
# ============================================================================
def get_user_doc(uid: str) -> Dict[str, Any]:
    snap = db.collection("users").document(uid).get()
    if not snap.exists:
        raise HTTPException(404, "User document not found")
    return snap.to_dict() or {}


def set_status(uid: str, status: str, error: str = ""):
    db.collection("users").document(uid).set(
        {"post_status": status, "last_error": error, "post_updatedAt": firestore.SERVER_TIMESTAMP},
        merge=True,
    )


def save_insight(uid: str, job_id: str, payload: Dict[str, Any]):
    db.collection("users").document(uid).collection("job_insights").document(job_id).set(payload, merge=True)


def set_post_fingerprint(uid: str, fp: Optional[str]):
    if fp:
        db.collection("users").document(uid).set({"system": {"post_fingerprint": fp}}, merge=True)


def save_ui_summary(uid: str, ui: Dict[str, Any]):
    db.collection("users").document(uid).set({"ui_summary": ui}, merge=True)

# ============================================================================
# JSON helpers
# ============================================================================
def _strip_code_fence(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def try_parse_json(text: str):
    if not text:
        return None
    for candidate in [text, _strip_code_fence(text)]:
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(candidate[start : end + 1])
        except Exception:
            pass
    return None


def to_plain(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))

# ============================================================================
# Normalizer (prediction 기반)
# ============================================================================
def _prio(node: Any, default: int = 4) -> int:
    """
    Firestore 필드가 dict({'priority':2})일 수도, 그냥 int/str일 수도 있음.
    priority가 없으면 default 사용. priority(1~6) -> weight(6~1)
    """
    try:
        if isinstance(node, dict):
            p = int(node.get("priority", default))
        else:
            p = default
    except Exception:
        p = default
    return max(1, 7 - p)


def normalize_user_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    pred = raw.get("prediction") or {}
    top5: List[str] = pred.get("top5") or []
    scores: Dict[str, float] = pred.get("scores") or {}

    recos: List[Dict[str, Any]] = [{"job": j, "score": scores.get(j)} for j in top5]

    weights = {
        "성격": _prio(raw.get("성격")),
        "흥미": _prio(raw.get("흥미")),
        "나이": _prio(raw.get("나이")),
        "성별": _prio(raw.get("성별")),
        "최종학력": _prio(None, default=4),
        "경제상황및대인": _prio(raw.get("경제상황 및 대인관계")),
        "감정기복": 3,
        "스트레스해소": _prio(raw.get("스트레스해소")),
        "관심분야": _prio(raw.get("관심분야")),
        "급여": _prio(raw.get("나이")),
    }

    user_profile = {
        "name": raw.get("이름"),
        "gender_code": (raw.get("성별") or {}).get("value"),
        "age": (raw.get("나이") or {}).get("value"),
        "min_expected_salary": raw.get("최소급여"),
    }
    answers = {
        "성격": raw.get("성격") or {},
        "흥미": raw.get("흥미") or {},
        "관심분야": raw.get("관심분야") or {},
        "경제상황 및 대인관계": raw.get("경제상황 및 대인관계") or {},
        "감정기복": raw.get("감정기복"),
        "스트레스해소": raw.get("스트레스해소") or {},
        "최종학력": raw.get("최종학력") or {},
    }

    return {
        "profile": to_plain(user_profile),
        "answers": to_plain(answers),
        "weights": to_plain(weights),
        "recommendations": to_plain(recos),
        "prediction": to_plain(pred),
    }

# ============================================================================
# Prompt builder (Gemini)
# ============================================================================
def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in (s or "")).strip("-")


def build_prompt(user_payload: Dict[str, Any], job_item: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    sys_text = (
        "당신은 진로 코치이자 커리어 컨설턴트입니다. 한국 취업/교육 환경 기준으로 "
        "과장 없이 실무적 조언을 하세요. 출력은 반드시 JSON으로 하고, 스키마를 준수하세요."
    )

    job_name = (job_item.get("job") or "").strip()
    job_id = _slug(job_name)

    model_input = {
        "user_profile": user_payload["profile"],
        "answers": user_payload["answers"],
        "weights": user_payload["weights"],
        "job_candidate": job_item,
        "notes": {"locale": "ko-KR", "style": "친절하고 명료, 실행 가능한 액션 중심"},
    }

    schema_json = json.dumps(JobInsight.model_json_schema(), indent=2, ensure_ascii=False)
    input_json = json.dumps(model_input, indent=2, ensure_ascii=False)

    content = [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        sys_text
                        + "\n\n"
                        f"[목표]\n'{job_name}' 직업에 대해 아래 3가지를 생성:\n"
                        "- analysis: 항목별 분석(성격/흥미/장점/리스크/액션)\n"
                        "- job_info: 직업 요약/핵심업무/핵심스킬/도구/연봉범주(선택)/전망(선택)/연관직무\n"
                        "- roadmap: 단계별 학습/경험 로드맵(**정확히 5단계**). 각 단계는 3~6개 구체적 과제로 구성하고, 가능하면 예상 기간(주)을 제시하세요.\n\n"
                        "[출력 스키마(JSON)]\n"
                        + schema_json
                        + "\n\n[입력 데이터]\n"
                        + input_json
                        + "\n\n[규칙] 결과는 JSON만 출력하세요. 설명/마크다운/코드펜스 금지."
                    )
                }
            ],
        }
    ]
    return content, job_id

def ensure_five_phases(rd: Roadmap) -> Roadmap:
    """모델 결과를 항상 5단계로 보정한다(부족하면 패딩, 많으면 잘라냄)."""
    phases = list(rd.phases or [])

    default_titles = [
        "1. 관련 학과·기초 다지기",
        "2. 기초 프로그래밍/도구 학습",
        "3. 실전 미니프로젝트",
        "4. 프레임워크/심화 역량",
        "5. 포트폴리오·실무 준비",
    ]

    if len(phases) > 5:
        phases = phases[:5]

    while len(phases) < 5:
        idx = len(phases)
        phases.append(
            RoadmapStep(
                title=default_titles[idx],
                tasks=["핵심 개념 학습", "실습 과제 수행"],
                resources=[],
                milestone=None,
                expected_duration_weeks=None,
            )
        )

    for ph in phases:
        if ph.tasks is None:
            ph.tasks = []
        if len(ph.tasks) < 3:
            ph.tasks += ["추가 학습", "간단 실습"][: max(0, 3 - len(ph.tasks))]

    rd.phases = phases
    return rd


# ============================================================================
# Gemini caller
# ============================================================================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_gemini(content: List[Dict[str, Any]]) -> Dict[str, Any]:
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(
        content,
        generation_config={
            "temperature": 0.3,
            "top_p": 0.9,
            "max_output_tokens": 1536,
            "response_mime_type": "application/json",
        },
    )

    text = getattr(resp, "text", None)
    if not text and getattr(resp, "candidates", None):
        parts = resp.candidates[0].content.parts
        if parts and getattr(parts[0], "text", None):
            text = parts[0].text

    parsed = try_parse_json(text or "")
    if parsed is not None:
        return parsed

    # 2차 보정 프롬프트
    schema_json = json.dumps(JobInsight.model_json_schema(), indent=2, ensure_ascii=False)
    fix_prompt = [
        {
            "role": "user",
            "parts": [
                {
                    "text": "아래 내용을 스키마에 맞춘 JSON으로만 출력하세요.\n\n[스키마]\n"
                    + schema_json
                    + "\n\n[고칠 원문]\n"
                    + (text or "")
                }
            ],
        }
    ]
    resp2 = model.generate_content(
        fix_prompt,
        generation_config={
            "temperature": 0.0,
            "top_p": 0.9,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
        },
    )
    text2 = getattr(resp2, "text", None)
    parsed2 = try_parse_json(text2 or "")
    if parsed2 is not None:
        return parsed2

    raise ValueError("Gemini JSON parse failed")

# ============================================================================
# UI Summary builders (rule-based)
# ============================================================================
def _pct(v: Optional[float], base: float) -> int:
    try:
        return max(0, min(100, int(round((float(v) / base) * 100))))
    except Exception:
        return 0

def _mean(nums: List[float]) -> float:
    arr = [float(x) for x in nums if x is not None]
    return sum(arr) / len(arr) if arr else 0.0


def build_big5_ui(ans: Dict[str, Any]) -> Dict[str, Any]:
    trait = ans.get("성격") or {}
    axes = [
        {"label": "개방성", "percent": _pct(trait.get("개방성"), 5)},
        {"label": "성실성", "percent": _pct(trait.get("성실성"), 5)},
        {"label": "외향성", "percent": _pct(trait.get("외향성"), 5)},
        {"label": "친화성", "percent": _pct(trait.get("친화성"), 5)},
        {"label": "신경성", "percent": _pct(trait.get("신경성"), 5)},
    ]
    bars = [UIValueBar(**a).model_dump() for a in axes]

    summary_parts = []
    if (trait.get("성실성") or 0) >= 4:
        summary_parts.append("책임감과 실행력이 높습니다")
    if (trait.get("개방성") or 0) >= 4:
        summary_parts.append("새로운 시도와 창의적 사고를 선호합니다")
    if (trait.get("외향성") or 0) >= 4:
        summary_parts.append("협업과 커뮤니케이션이 강점입니다")
    if (trait.get("친화성") or 0) >= 4:
        summary_parts.append("관계 형성에 유리합니다")
    if (trait.get("신경성") or 3) <= 2:
        summary_parts.append("정서적으로 안정적인 편입니다")
    summary = " · ".join(summary_parts) if summary_parts else "균형적인 성향입니다"

    return {
        "radar": UIRadar(axes=[UIValueBar(**a) for a in axes]).model_dump(),
        "bars": bars,
        "summary": summary,
    }


def build_interests_ui(ans: Dict[str, Any], trait: Dict[str, Any]) -> Dict[str, Any]:
    inter = ans.get("흥미") or {}
    # 탐구형은 개방성/성실성 평균으로 추정
    inv = int(round((_mean([trait.get("개방성", 3), trait.get("성실성", 3)]) / 5) * 100))
    axes = [
        {"label": "예술형", "percent": _pct(inter.get("예술형"), 5)},
        {"label": "사회형", "percent": _pct(inter.get("사회형"), 5)},
        {"label": "실재형", "percent": _pct(inter.get("실재형"), 5)},
        {"label": "탐구형", "percent": inv},
        {"label": "기업형", "percent": _pct(inter.get("기업형"), 5)},
        {"label": "관습형", "percent": _pct(inter.get("관습형"), 5)},
    ]
    kw = {
        "예술형": ["창의적", "직관적", "감성적"],
        "사회형": ["친근함", "협력적", "이해심"],
        "실재형": ["실용적", "현실적", "손끝재주"],
        "탐구형": ["분석적", "논리적", "호기심"],
        "기업형": ["적극적", "야심적", "사교적"],
        "관습형": ["정확함", "신중함", "체계적"],
    }
    cards = [{"name": a["label"], "percent": a["percent"], "keywords": kw[a["label"]]} for a in axes]
    top = sorted(axes, key=lambda x: x["percent"], reverse=True)[:2]
    summary = f"{top[0]['label']}·{top[1]['label']} 성향이 강하며, 해당 영역의 학습/경험에서 동기가 높습니다."
    return {"radar": UIRadar(axes=[UIValueBar(**a) for a in axes]).model_dump(), "cards": cards, "summary": summary}


def build_affect_values_ui(ans: Dict[str, Any], trait: Dict[str, Any]) -> Dict[str, Any]:
    mood = ans.get("감정기복")                      # 1~4 정수
    econ = ans.get("경제상황 및 대인관계") or {}

    # 스트레스해소는 1~4 정수; 혹시 dict 형태여도 안전 처리
    stress_raw = ans.get("스트레스해소")
    stress = stress_raw.get("value") if isinstance(stress_raw, dict) else stress_raw

    ach = _pct(trait.get("성실성", 3), 5)  # 성취
    autonomy = _pct(_mean([trait.get("개방성", 3), trait.get("외향성", 3)]), 5)  # 자율성
    # 감정기복(1~4, 작을수록 안정) → 안정성(%)로 역변환
    stability = int(round(((5 - (mood or 3)) / 4) * 100))

    relation = _pct(_mean([trait.get("친화성", 3), econ.get("대인관계", 3)]), 5)
    creativity = _pct(_mean([trait.get("개방성", 3)]), 5)

    top_values = [
        {"name": "성취", "percent": ach},
        {"name": "자율성", "percent": autonomy},
        {"name": "안정성", "percent": stability},
        {"name": "관계", "percent": relation},
        {"name": "창의성", "percent": creativity},
    ]
    top_values = sorted(top_values, key=lambda x: x["percent"], reverse=True)[:3]

    relationship_mood = _pct(_mean([trait.get("친화성", 3), trait.get("외향성", 3)]), 5)
    stress_relief = _pct(stress or 0, 4)  # 1~4 기준

    summary = "핵심 가치로 " + ", ".join([f"{v['name']}({v['percent']}%)" for v in top_values]) + "가 두드러집니다."
    return {
        "top_values": top_values,
        "relationship_mood_percent": relationship_mood,
        "stress_relief_percent": stress_relief,
        "summary": summary,
    }

def build_overall_ui(big5: Dict[str, Any], interests: Dict[str, Any], affect: Dict[str, Any]) -> Dict[str, Any]:
    insights = [
        "성격적 강점을 커리어 목표와 연결하여 성장 속도를 높이세요.",
        "관심 높은 영역에서 실전 프로젝트 경험을 쌓으면 동기와 성취가 커집니다.",
        "가치관 상위 항목을 만족할 수 있는 업무 환경을 우선 고려하세요.",
    ]
    core_values_top3 = affect["top_values"]
    growth_areas = ["기술적 깊이 확장", "체계적 업무 관리", "비즈니스 감각"]
    traits = ["높은 성실성과 책임감", "협업 능력 우수", "새로운 시도에 개방적"]
    summary = (
        "종합적으로 강점을 바탕으로 실무 경험을 확장하고, 단계적 로드맵을 통해 목표 직무 역량을 체계적으로 강화하세요."
    )
    return {
        "insights": insights,
        "core_values_top3": core_values_top3,
        "growth_areas": growth_areas,
        "traits": traits,
        "summary": summary,
    }


def build_career_cards(top5: List[str], insights_by_job: Dict[str, JobInsight]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for job in top5:
        ins = insights_by_job.get(job)
        sub = (ins.job_info.summary if ins else "") or ""
        tags = (ins.job_info.core_skills[:3] if ins else []) or []
        salary = ins.job_info.average_salary_krw if ins else None
        subtitle = sub[:40].rstrip() + ("…" if len(sub) > 40 else "")
        cards.append(UICareerCard(job=job, subtitle=subtitle, tags=tags, avg_salary_text=salary).model_dump())
    return cards


def build_roadmap_blocks(roadmap: Roadmap) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for ph in roadmap.phases:
        items = [{"text": t, "done": False} for t in ph.tasks]
        base = 0
        if ph.resources:
            base += 10
        if ph.milestone:
            base += 20
        blocks.append(UIRoadmapBlock(title=ph.title, items=items, progress=base).model_dump())
    return blocks


def build_ui_summary(raw: Dict[str, Any], pred: Dict[str, Any], insights: Dict[str, JobInsight]) -> Dict[str, Any]:
    ans = {
        "성격": raw.get("성격") or {},
        "흥미": raw.get("흥미") or {},
        "감정기복": raw.get("감정기복"),
        "스트레스해소": raw.get("스트레스해소"),        # ✅ 정수 그대로
        "경제상황 및 대인관계": raw.get("경제상황 및 대인관계") or {},
    }
    trait = ans["성격"]
    big5 = build_big5_ui(ans)
    interests = build_interests_ui(ans, trait)
    affect = build_affect_values_ui(ans, trait)
    overall = build_overall_ui(big5, interests, affect)

    top5 = pred.get("top5") or []
    cards = build_career_cards(top5, insights)
    roadmaps: Dict[str, List[Dict[str, Any]]] = {}
    for job in top5:
        ins = insights.get(job)
        if not ins:
            continue
        rid = _slug(job)
        roadmaps[rid] = build_roadmap_blocks(ins.roadmap)

    return UISummary(
        big5=big5,
        interests=interests,
        affect_values=affect,
        overall=overall,
        career_options=[UICareerCard(**c) for c in cards],
        roadmaps=roadmaps,
    ).model_dump()

# ============================================================================
# Endpoints / Core
# ============================================================================
@router.post("/process/{uid}")
def process_user(uid: str, background: BackgroundTasks):
    background.add_task(_process_user_sync, uid)
    return {"ok": True, "message": "processing in background"}


def _process_user_sync(uid: str):
    raw = get_user_doc(uid)
    pred = raw.get("prediction") or {}
    top5 = pred.get("top5") or []
    if not top5:
        raise ValueError("prediction.top5 is empty")

    set_status(uid, "processing")

    insights_by_job: Dict[str, JobInsight] = {}
    try:
        payload = normalize_user_payload(raw)

        # 1) 직업별 인사이트 + 로드맵 생성/저장
        for item in payload["recommendations"]:
            job = item.get("job")
            if not job:
                continue
            content, job_id = build_prompt(payload, item)
            try:
                out = call_gemini(content)
                insight = JobInsight(**out)
                # 항상 5단계로 보정
                insight.roadmap = ensure_five_phases(insight.roadmap)
                save_insight(uid, job_id, insight.model_dump())
                insights_by_job[job] = insight
            except Exception as e:
                # 직업별 에러는 문서로 남기고 계속 진행
                save_insight(
                    uid,
                    job_id,
                    {
                        "error": str(e),
                        "model": GEMINI_MODEL,
                        "updatedAt": datetime.datetime.utcnow().isoformat() + "Z",
                    },
                )

        # 2) 프론트 전용 요약 저장
        ui = build_ui_summary(raw, pred, insights_by_job)
        save_ui_summary(uid, to_plain(ui))

        # 3) 처리 지문 마킹 및 상태 완료
        set_post_fingerprint(uid, (raw.get("system") or {}).get("input_fingerprint"))
        set_status(uid, "done")

    except Exception as e:
        # 전체 파이프라인 실패 시
        set_status(uid, "error", str(e))
        raise


@router.get("/insights/{uid}")
def get_insights(uid: str):
    col = db.collection("users").document(uid).collection("job_insights").stream()
    items = [{"job_id": d.id, **(d.to_dict() or {})} for d in col]
    return {"count": len(items), "items": items}
