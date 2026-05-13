"""
기숙사 퇴사 점검 — Gemini VLM 분석 서비스.

기존 Streamlit 2_Student.py 의 vlm_analyze() 를 FastAPI 서비스 레이어로 분리합니다.
API 키는 st.secrets 가 아닌 환경변수(GEMINI_API_KEY)에서만 읽습니다.

주요 설계 결정:
  - analyze_closeup()은 동기 함수로 구현합니다.
    FastAPI endpoint 에서는 run_in_threadpool 또는 asyncio.to_thread 로 감쌉니다.
  - GeminiConfigError (키 미설정)를 제외한 모든 실패는 result="suspicious" fallback을 반환합니다.
    애플리케이션이 Gemini 오류로 죽지 않도록 합니다.
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

# ────────────────────────────────────────────────────────────
# 프롬프트 (Streamlit 원본 의미 보존)
# ────────────────────────────────────────────────────────────

_VLM_PROMPT = """당신은 기숙사 퇴사 점검 AI입니다.
아래 클로즈업 사진을 보고, 해당 부위가 원상복구가 필요한 상태인지 판단해 주세요.

판단 기준:
- 쓰레기, 이물질, 오염, 낙서, 파손, 비품 분실 또는 추가 물품 등이 있으면 "suspicious"
- 깨끗하고 정상 상태이면 "clean"

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{"result": "clean" or "suspicious", "reason": "한국어로 간단한 사유"}"""

_FALLBACK_REASON = "AI 분석 결과를 안정적으로 해석하지 못해 관리자 확인이 필요합니다."

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
    """analyze_closeup() 반환 값."""
    result: Literal["clean", "suspicious"]
    reason: str
    raw_text: str | None = field(default=None)


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

    Parameters
    ----------
    text : str
        Gemini API 응답 텍스트.

    Returns
    -------
    dict
        파싱된 JSON 오브젝트.

    Raises
    ------
    GeminiAnalysisError
        유효한 JSON 오브젝트를 찾지 못한 경우.
    """
    # 1. 코드 펜스 제거
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.MULTILINE).strip()

    # 2. 전체 텍스트 파싱 시도
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. 첫 번째 '{...}' 블록 추출 후 재시도
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
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
# 메인 함수 (동기)
# ────────────────────────────────────────────────────────────

def analyze_closeup(
    image_path: str | None = None,
    image_bytes: bytes | None = None,
) -> GeminiAnalysisResult:
    """
    클로즈업 이미지를 Gemini로 분석하여 clean/suspicious 판정 결과를 반환합니다.

    Parameters
    ----------
    image_path : str | None
        DB 에 저장된 이미지 파일명 (또는 절대경로). storage_service.load_image_bgr 로 로드합니다.
    image_bytes : bytes | None
        업로드된 이미지 raw bytes. image_path 와 둘 중 하나를 반드시 전달해야 합니다.

    Returns
    -------
    GeminiAnalysisResult
        result="clean"|"suspicious", reason(한국어), raw_text(Gemini 원문).

    Raises
    ------
    GeminiConfigError
        GEMINI_API_KEY 가 설정되지 않은 경우. 이 예외는 fallback 처리하지 않습니다.
    ValueError
        image_path 와 image_bytes 가 모두 None 인 경우.

    Notes
    -----
    - GeminiConfigError / ValueError 를 제외한 모든 오류는 result="suspicious" fallback 으로 처리됩니다.
    - FastAPI endpoint 에서 호출 시 asyncio.to_thread 또는 run_in_threadpool 로 감싸세요.
    """
    # ── 설정 확인 ────────────────────────────────────────────
    api_key = settings.GEMINI_API_KEY
    if not api_key:
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

    # ── Gemini API 호출 ──────────────────────────────────────
    raw_text: str | None = None
    try:
        client = genai.Client(api_key=api_key)
        image_part = genai_types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[image_part, _VLM_PROMPT],
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

    # ── 결과 정규화 ───────────────────────────────────────────
    result_val = parsed.get("result", "")
    if result_val not in ("clean", "suspicious"):
        logger.warning(
            "Gemini result 값이 clean/suspicious 가 아님(%r), suspicious fallback 적용", result_val
        )
        result_val = "suspicious"

    reason_val = parsed.get("reason", "").strip()
    if not reason_val:
        reason_val = _FALLBACK_REASON

    return GeminiAnalysisResult(
        result=result_val,  # type: ignore[arg-type]
        reason=reason_val,
        raw_text=raw_text,
    )


# ────────────────────────────────────────────────────────────
# async 래퍼 (FastAPI endpoint 에서 직접 사용)
# ────────────────────────────────────────────────────────────

async def analyze_closeup_async(
    image_path: str | None = None,
    image_bytes: bytes | None = None,
) -> GeminiAnalysisResult:
    """
    analyze_closeup() 의 async 래퍼.

    Gemini SDK 는 동기 API 이므로 asyncio.to_thread 로 스레드풀에서 실행합니다.
    FastAPI endpoint 에서 ``await analyze_closeup_async(...)`` 형태로 사용하세요.

    TODO (Phase 6 이후):
      - httpx AsyncClient 기반 비동기 Gemini 클라이언트로 교체하면
        스레드풀 없이 네이티브 async 가 가능합니다.
    """
    return await asyncio.to_thread(
        analyze_closeup,
        image_path=image_path,
        image_bytes=image_bytes,
    )
