"""
기숙사 퇴사 점검 — Gemini VLM 분석 서비스.

설계 철학: VLM은 최종 판정자가 아니라 관리자를 위한 확인 보조자입니다.
  - 학생이 제출한 근접 촬영 이미지와 메모(student_note)를 함께 분석합니다.
  - VLM은 "학생 설명이 이미지상 타당한지"를 검토하여 참고 의견을 제공합니다.
  - 손상 여부를 단정짓지 않으며, 관리자의 최종 판단을 보조합니다.

주요 설계 결정:
  - analyze_closeup()은 동기 함수로 구현합니다.
    FastAPI endpoint 에서는 asyncio.to_thread 로 감쌉니다.
  - GeminiConfigError(키 미설정)를 제외한 모든 실패는 result="suspicious" fallback을 반환합니다.
  - analyze_closeup_async() async 래퍼를 함께 제공합니다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from google import genai
from google.genai import types as genai_types

from app.core.config import settings
from app.services.storage_service import (
    decode_image_bytes_to_bgr,
    encode_bgr_to_jpeg_bytes,
    load_image_bgr,
)

logger = logging.getLogger(__name__)

_PLACEHOLDER_API_KEYS = {
    "your_gemini_api_key_here",
    "YOUR_GEMINI_API_KEY",
    "your_api_key_here",
}

_FALLBACK_REASON = "AI 분석 결과를 안정적으로 해석하지 못해 관리자 직접 확인이 필요합니다."


# ────────────────────────────────────────────────────────────
# 프롬프트 빌더
# ────────────────────────────────────────────────────────────

def _build_prompt(student_note: str | None) -> str:
    """
    student_note 유무에 따라 VLM 프롬프트를 동적으로 생성합니다.

    student_note가 있을 때:
      - 학생 설명을 학생의 "주장/해명"으로 간주합니다.
      - 이미지를 먼저 독립적으로 관찰한 후, 학생 설명과 대조합니다.
      - 일치 여부를 note_consistency 값으로 표시합니다.

    student_note가 없을 때:
      - 이미지 단독으로 시각적 특징을 중립적으로 기술합니다.
    """
    json_schema = """{
  "result": "clean" 또는 "suspicious",
  "note_consistency": "consistent" 또는 "partially_consistent" 또는 "inconsistent" 또는 "unclear",
  "visual_observation": "근접 이미지에서 실제로 관찰되는 내용",
  "student_note_assessment": "학생 설명과 이미지 관찰 결과 비교",
  "possible_explanation": "학생 설명이 충분히 맞지 않을 경우 이미지상 대안 설명 (해당 없으면 null)",
  "review_recommendation": "관리자가 최종 판단 시 확인할 포인트",
  "reason": "위 내용을 종합한 한국어 요약 (2~3문장, 중립적 표현)"
}"""

    if student_note:
        return f"""당신은 기숙사 퇴사 점검 보조 AI입니다.
아래 근접 촬영 이미지와 학생이 제출한 설명을 함께 검토하고,
관리자의 최종 판단을 위한 참고 의견을 제공해 주세요.

[학생 설명]
"{student_note}"

[검토 방법]
1. 근접 이미지를 독립적으로 먼저 관찰하고 시각적 특징을 파악하세요.
2. 이미지에서 보이는 내용을 기준으로 학생 설명이 타당한지 확인하세요.
3. 학생 설명과 이미지가 일치하면 "consistent", 일부만 맞으면 "partially_consistent",
   맞지 않으면 "inconsistent", 판단하기 어려우면 "unclear"로 표시하세요.

[중요 원칙]
- 손상 여부를 단정짓지 마세요.
- "가능성", "관찰됨", "추가 확인 필요", "~처럼 보임" 등 중립적 표현만 사용하세요.
- "학생 책임", "파손 확정", "수리비 청구" 같은 표현은 절대 사용하지 마세요.
- 이 의견은 관리자 최종 판단을 위한 참고 자료입니다.

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{json_schema}"""
    else:
        return f"""당신은 기숙사 퇴사 점검 보조 AI입니다.
아래 근접 촬영 이미지를 보고, 관리자의 최종 판단을 위한 참고 의견을 제공해 주세요.
학생이 별도 설명을 남기지 않았습니다.

[검토 방법]
이미지에서 관찰되는 시각적 특징(표면 상태, 색상 변화, 오염, 질감 등)을 중립적으로 기술하세요.

[중요 원칙]
- 손상 여부를 단정짓지 마세요.
- "가능성", "관찰됨", "추가 확인 필요", "~처럼 보임" 등 중립적 표현만 사용하세요.
- "학생 책임", "파손 확정", "수리비 청구" 같은 표현은 절대 사용하지 마세요.

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{json_schema}"""


# ────────────────────────────────────────────────────────────
# 커스텀 예외
# ────────────────────────────────────────────────────────────

class GeminiServiceError(RuntimeError):
    """Gemini 서비스 전반의 기본 예외."""


class GeminiConfigError(GeminiServiceError):
    """GEMINI_API_KEY 미설정 등 설정 오류."""


class GeminiAnalysisError(GeminiServiceError):
    """API 호출 또는 응답 파싱 실패."""


# ────────────────────────────────────────────────────────────
# 결과 데이터 클래스
# ────────────────────────────────────────────────────────────

@dataclass
class GeminiAnalysisResult:
    """
    analyze_closeup() 반환 값.

    reason 은 vlm_reason DB 컬럼에 저장되는 사람이 읽기 좋은 형태의 참고 의견입니다.
    구조화된 필드(note_consistency 등)는 호출부에서 필요시 활용할 수 있습니다.
    """
    result: Literal["clean", "suspicious"]
    reason: str                          # DB 저장 및 화면 표시용 요약 텍스트
    raw_text: str | None = field(default=None)
    # 구조화된 VLM 참고 의견 (선택적)
    note_consistency: str | None = field(default=None)
    visual_observation: str | None = field(default=None)
    student_note_assessment: str | None = field(default=None)
    possible_explanation: str | None = field(default=None)
    review_recommendation: str | None = field(default=None)


# ────────────────────────────────────────────────────────────
# JSON 파싱 헬퍼
# ────────────────────────────────────────────────────────────

def parse_gemini_json_response(text: str) -> dict:
    """
    Gemini 응답 텍스트에서 JSON 오브젝트를 추출합니다.

    처리 순서:
    1. 마크다운 코드 펜스 제거 (```json ... ```)
    2. 전체 텍스트를 json.loads 시도
    3. 실패 시 첫 번째 '{...}' 블록을 찾아 재시도

    Raises
    ------
    GeminiAnalysisError
        유효한 JSON 오브젝트를 찾지 못한 경우.
    """
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise GeminiAnalysisError(
        f"Gemini 응답에서 유효한 JSON 오브젝트를 찾지 못했습니다. "
        f"원문(앞 200자): {text[:200]!r}"
    )


# ────────────────────────────────────────────────────────────
# 결과 포맷 헬퍼
# ────────────────────────────────────────────────────────────

_CONSISTENCY_LABELS: dict[str, str] = {
    "consistent":           "학생 설명과 이미지가 대체로 일치함",
    "partially_consistent": "학생 설명과 이미지가 일부 일치함",
    "inconsistent":         "학생 설명과 이미지가 일치하지 않음",
    "unclear":              "이미지만으로 판단하기 어려움",
}


def _format_reason(parsed: dict, has_student_note: bool) -> str:
    """
    Gemini 파싱 결과를 관리자가 읽기 좋은 다단 텍스트로 변환합니다.

    vlm_reason DB 컬럼에 저장되어 관리자 화면에 표시됩니다.
    """
    lines: list[str] = []

    # ── Gemini 자체 요약 문장 ──────────────────────────────────
    summary = (parsed.get("reason") or "").strip()
    if summary:
        lines.append(summary)

    # ── 구조화 정보 (구분선 이후) ─────────────────────────────
    details: list[str] = []

    visual = (parsed.get("visual_observation") or "").strip()
    if visual:
        details.append(f"▸ 이미지 관찰: {visual}")

    if has_student_note:
        consistency_key = (parsed.get("note_consistency") or "unclear").strip()
        consistency_label = _CONSISTENCY_LABELS.get(consistency_key, consistency_key)
        details.append(f"▸ 학생 설명 검토: {consistency_label}")

        assessment = (parsed.get("student_note_assessment") or "").strip()
        if assessment:
            details.append(f"  {assessment}")

        alt_explanation = (parsed.get("possible_explanation") or "").strip()
        if alt_explanation and alt_explanation.lower() != "null":
            details.append(f"▸ 대안 설명: {alt_explanation}")

    recommendation = (parsed.get("review_recommendation") or "").strip()
    if recommendation:
        details.append(f"▸ 확인 포인트: {recommendation}")

    if details:
        if lines:
            lines.append("")  # 빈 줄 구분
        lines.extend(details)

    return "\n".join(lines) if lines else _FALLBACK_REASON


# ────────────────────────────────────────────────────────────
# 메인 함수 (동기)
# ────────────────────────────────────────────────────────────

def analyze_closeup(
    image_path: str | None = None,
    image_bytes: bytes | None = None,
    student_note: str | None = None,
) -> GeminiAnalysisResult:
    """
    근접 촬영 이미지와 학생 메모를 Gemini로 분석하여 참고 의견을 반환합니다.

    Parameters
    ----------
    image_path : str | None
        DB 에 저장된 이미지 파일명 (또는 절대경로).
    image_bytes : bytes | None
        업로드된 이미지 raw bytes. image_path 와 둘 중 하나를 반드시 전달해야 합니다.
    student_note : str | None
        학생이 근접 촬영 시 남긴 메모. 제공되면 이미지와 대조 분석합니다.

    Returns
    -------
    GeminiAnalysisResult
        result, reason(관리자용 참고 의견), 구조화된 필드들.

    Raises
    ------
    GeminiConfigError
        GEMINI_API_KEY 가 설정되지 않은 경우.
    ValueError
        image_path 와 image_bytes 가 모두 None 인 경우.
    """
    # ── 설정 확인 ────────────────────────────────────────────
    api_key = (settings.GEMINI_API_KEY or "").strip()
    if not api_key or api_key in _PLACEHOLDER_API_KEYS:
        raise GeminiConfigError(
            "GEMINI_API_KEY 환경변수가 설정되지 않았습니다. "
            "backend/.env 파일에 GEMINI_API_KEY=... 를 추가하세요."
        )

    if image_path is None and image_bytes is None:
        raise ValueError("image_path 또는 image_bytes 중 하나를 반드시 전달해야 합니다.")

    # ── 이미지 로드 ──────────────────────────────────────────
    try:
        if image_bytes is not None:
            bgr = decode_image_bytes_to_bgr(image_bytes)
        else:
            bgr = load_image_bgr(image_path)  # type: ignore[arg-type]
    except Exception as exc:
        logger.warning("이미지 로드 실패, suspicious fallback 반환: %s", exc)
        return GeminiAnalysisResult(result="suspicious", reason=_FALLBACK_REASON)

    # ── BGR → JPEG bytes ─────────────────────────────────────
    try:
        jpeg_bytes = encode_bgr_to_jpeg_bytes(bgr, quality=90)
    except Exception as exc:
        logger.warning("이미지 JPEG 인코딩 실패, suspicious fallback 반환: %s", exc)
        return GeminiAnalysisResult(result="suspicious", reason=_FALLBACK_REASON)

    # ── 프롬프트 구성 ─────────────────────────────────────────
    note = student_note.strip() if student_note else None
    prompt = _build_prompt(note)

    # ── Gemini API 호출 ──────────────────────────────────────
    raw_text: str | None = None
    try:
        client = genai.Client(api_key=api_key)
        image_part = genai_types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[image_part, prompt],
        )
        raw_text = response.text
    except Exception as exc:
        logger.warning("Gemini API 호출 실패, suspicious fallback 반환: %s", exc)
        return GeminiAnalysisResult(
            result="suspicious", reason=_FALLBACK_REASON, raw_text=None
        )

    # ── JSON 파싱 ─────────────────────────────────────────────
    try:
        parsed = parse_gemini_json_response(raw_text)
    except GeminiAnalysisError as exc:
        logger.warning("Gemini 응답 JSON 파싱 실패, suspicious fallback 반환: %s", exc)
        return GeminiAnalysisResult(
            result="suspicious", reason=_FALLBACK_REASON, raw_text=raw_text
        )

    # ── result 정규화 ──────────────────────────────────────────
    result_val = parsed.get("result", "")
    if result_val not in ("clean", "suspicious"):
        logger.warning("Gemini result 값이 clean/suspicious 가 아님(%r), suspicious fallback 적용", result_val)
        result_val = "suspicious"

    # ── 구조화 필드 추출 ───────────────────────────────────────
    note_consistency      = (parsed.get("note_consistency") or "unclear").strip()
    visual_observation    = (parsed.get("visual_observation") or "").strip() or None
    student_note_assess   = (parsed.get("student_note_assessment") or "").strip() or None
    possible_explanation  = (parsed.get("possible_explanation") or "").strip() or None
    if possible_explanation and possible_explanation.lower() == "null":
        possible_explanation = None
    review_recommendation = (parsed.get("review_recommendation") or "").strip() or None

    # ── 사람이 읽기 좋은 reason 생성 ─────────────────────────
    reason_val = _format_reason(parsed, has_student_note=bool(note))
    if not reason_val:
        reason_val = _FALLBACK_REASON

    return GeminiAnalysisResult(
        result=result_val,  # type: ignore[arg-type]
        reason=reason_val,
        raw_text=raw_text,
        note_consistency=note_consistency,
        visual_observation=visual_observation,
        student_note_assessment=student_note_assess,
        possible_explanation=possible_explanation,
        review_recommendation=review_recommendation,
    )


# ────────────────────────────────────────────────────────────
# async 래퍼 (FastAPI endpoint 에서 직접 사용)
# ────────────────────────────────────────────────────────────

async def analyze_closeup_async(
    image_path: str | None = None,
    image_bytes: bytes | None = None,
    student_note: str | None = None,
) -> GeminiAnalysisResult:
    """
    analyze_closeup() 의 async 래퍼.

    Gemini SDK 는 동기 API 이므로 asyncio.to_thread 로 스레드풀에서 실행합니다.
    FastAPI endpoint 에서 ``await analyze_closeup_async(...)`` 형태로 사용하세요.
    """
    return await asyncio.to_thread(
        analyze_closeup,
        image_path=image_path,
        image_bytes=image_bytes,
        student_note=student_note,
    )
