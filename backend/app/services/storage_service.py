"""
이미지 저장/로드 서비스.

역할:
  - UploadFile / bytes / BGR ndarray 를 data/images/ 에 JPEG 로 저장
  - DB 에 저장할 파일명(relative_path)과 frontend 에 반환할 public_url 을 생성
  - 기존 Streamlit db_utils.py 의 이미지 I/O 함수를 FastAPI 환경으로 대체

기존 Streamlit 함수 대응:
  save_image_bytes(room_id, type, data)   → save_image_bytes(data, type, room_number=...)
  save_image_ndarray(room_id, type, img)  → save_ndarray(img, type, room_number=...)
  load_image(path)                        → load_image_bgr(path)

Windows 한글 경로 대응:
  cv2.imread / cv2.imwrite 는 한글 경로를 처리하지 못합니다.
  모든 read/write 는 np.fromfile+cv2.imdecode / cv2.imencode+open 방식을 사용합니다.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile

from app.core.config import settings
from app.core.constants import ImageType


# ── 저장 결과 DTO ───────────────────────────────────────────────────────────

@dataclass
class StoredImage:
    """
    이미지 저장 후 반환되는 결과 객체.

    Attributes
    ----------
    filename      : 저장된 파일명  (예: "101_initial_20260513_153012_ab12cd.jpg")
    file_path     : 디스크 절대 경로 (내부 용도)
    relative_path : 스토리지 루트 기준 상대 경로 = filename  (DB 저장용)
    public_url    : frontend 에서 접근 가능한 전체 URL
                    (예: "http://localhost:8000/images/101_initial_...jpg")
    content_type  : MIME 타입 ("image/jpeg")
    size_bytes    : 저장된 파일 크기 (bytes)
    """
    filename: str
    file_path: Path
    relative_path: str
    public_url: str
    content_type: str
    size_bytes: int


# ── 내부 헬퍼 ───────────────────────────────────────────────────────────────

def _get_storage_dir() -> Path:
    """스토리지 디렉터리를 반환합니다. 없으면 생성합니다."""
    d = settings.image_storage_path
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_filename(
    image_type: str,
    room_number: str | None = None,
    inspection_id: int | None = None,
    issue_id: int | None = None,
    extension: str = ".jpg",
) -> str:
    """
    충돌 없는 파일명을 생성합니다.

    우선순위: issue_id > inspection_id > room_number > "unknown"

    형식:
      {prefix}_{image_type}_{YYYYMMDD_HHMMSS}_{uuid6}{ext}

    예:
      issue_7_closeup_20260513_153012_ab12cd.jpg
      inspection_3_final_20260513_153012_ab12cd.jpg
      101_initial_20260513_153012_ab12cd.jpg
    """
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:6]

    if issue_id is not None:
        prefix = f"issue_{issue_id}"
    elif inspection_id is not None:
        prefix = f"inspection_{inspection_id}"
    elif room_number is not None:
        # 슬래시, 공백 등 경로에 안전하지 않은 문자를 제거
        safe_num = room_number.replace("/", "-").replace(" ", "_")
        prefix = safe_num
    else:
        prefix = "unknown"

    return f"{prefix}_{image_type}_{ts}_{short_id}{extension}"


# ── Public URL 변환 ─────────────────────────────────────────────────────────

def get_public_image_url(path_or_filename: str) -> str:
    """
    파일명 또는 절대 경로로부터 public URL 을 생성합니다.

    FastAPI StaticFiles 는 /images 경로로 마운트되며,
    BASE_URL 설정값(기본: http://localhost:8000)을 사용합니다.

    Examples
    --------
    get_public_image_url("101_initial_xxx.jpg")
        → "http://localhost:8000/images/101_initial_xxx.jpg"

    get_public_image_url("/abs/path/to/data/images/101_initial_xxx.jpg")
        → "http://localhost:8000/images/101_initial_xxx.jpg"
    """
    filename = Path(path_or_filename).name
    base = settings.BASE_URL.rstrip("/")
    return f"{base}/images/{filename}"


# ── 저장 함수 ───────────────────────────────────────────────────────────────

async def save_upload_file(
    file: UploadFile,
    image_type: str,
    room_number: str | None = None,
    inspection_id: int | None = None,
    issue_id: int | None = None,
) -> StoredImage:
    """
    FastAPI UploadFile 을 읽어 JPEG 파일로 저장합니다.

    Parameters
    ----------
    file          : `File(...)` 로 받은 UploadFile
    image_type    : ImageType 상수 (예: ImageType.INITIAL)
    room_number   : 호실 번호 문자열 (파일명 prefix 용)
    inspection_id : Inspection 내부 ID
    issue_id      : Issue 내부 ID

    Returns
    -------
    StoredImage
    """
    image_bytes = await file.read()
    return save_image_bytes(
        image_bytes,
        image_type,
        room_number=room_number,
        inspection_id=inspection_id,
        issue_id=issue_id,
    )


def save_image_bytes(
    image_bytes: bytes,
    image_type: str,
    room_number: str | None = None,
    inspection_id: int | None = None,
    issue_id: int | None = None,
    extension: str = ".jpg",
) -> StoredImage:
    """
    bytes(JPEG / PNG) 를 파일로 저장합니다.

    기존 Streamlit 의 save_image_bytes(room_id, image_type, data) 에 대응합니다.

    Parameters
    ----------
    image_bytes : 파일 바이트 (st.file_uploader 또는 UploadFile.read() 결과)
    extension   : 저장 확장자 (기본 ".jpg")

    Returns
    -------
    StoredImage
    """
    if not image_bytes:
        raise ValueError("저장할 image_bytes 가 비어 있습니다.")

    storage_dir = _get_storage_dir()
    filename    = _make_filename(image_type, room_number, inspection_id, issue_id, extension)
    file_path   = storage_dir / filename

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    return StoredImage(
        filename      = filename,
        file_path     = file_path,
        relative_path = filename,
        public_url    = get_public_image_url(filename),
        content_type  = "image/jpeg",
        size_bytes    = len(image_bytes),
    )


def save_ndarray(
    image: np.ndarray,
    image_type: str,
    room_number: str | None = None,
    inspection_id: int | None = None,
    issue_id: int | None = None,
    quality: int = 90,
) -> StoredImage:
    """
    OpenCV BGR ndarray 를 JPEG 로 인코딩하여 저장합니다.

    기존 Streamlit 의 save_image_ndarray(room_id, image_type, img) 에 대응합니다.

    Parameters
    ----------
    image   : BGR ndarray (uint8)
    quality : JPEG 압축 품질 (1–100, 기본 90)

    Raises
    ------
    ValueError  : image 가 None 이거나 비어 있을 때

    Returns
    -------
    StoredImage
    """
    if image is None or image.size == 0:
        raise ValueError("저장할 image 가 None 이거나 비어 있습니다.")

    jpeg_bytes = encode_bgr_to_jpeg_bytes(image, quality)
    return save_image_bytes(
        jpeg_bytes,
        image_type,
        room_number=room_number,
        inspection_id=inspection_id,
        issue_id=issue_id,
    )


# ── 로드 / 디코드 함수 ──────────────────────────────────────────────────────

def load_image_bgr(path: str | Path) -> np.ndarray:
    """
    디스크의 이미지 파일을 BGR ndarray 로 읽습니다.

    cv2.imread 를 직접 사용하지 않고 np.fromfile + cv2.imdecode 를 사용하여
    Windows 한글 경로 문제를 회피합니다.

    기존 Streamlit 의 load_image(path) 에 대응합니다.

    Parameters
    ----------
    path : 절대 경로 또는 스토리지 디렉터리 기준 상대 경로

    Raises
    ------
    FileNotFoundError : 파일이 존재하지 않을 때
    ValueError        : 이미지 디코딩에 실패했을 때

    Returns
    -------
    np.ndarray : BGR uint8 이미지
    """
    p = Path(path)

    # 상대 경로로 전달된 경우 스토리지 디렉터리 기준으로 해석
    if not p.is_absolute():
        p = _get_storage_dir() / p

    if not p.exists():
        raise FileNotFoundError(f"이미지 파일이 없습니다: {p}")

    buf = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지 디코딩에 실패했습니다: {p}")
    return img


def decode_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """
    bytes (JPEG / PNG) 를 BGR ndarray 로 디코딩합니다.

    기존 Streamlit 의 decode_bytes() / decode_uploaded_image() 에 대응합니다.

    Parameters
    ----------
    image_bytes : JPEG 또는 PNG 바이트

    Raises
    ------
    ValueError : decode 실패 시

    Returns
    -------
    np.ndarray : BGR uint8 이미지
    """
    if not image_bytes:
        raise ValueError("디코딩할 image_bytes 가 비어 있습니다.")

    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "image_bytes 디코딩에 실패했습니다. "
            "유효한 JPEG / PNG 데이터인지 확인하세요."
        )
    return img


# ── 인코드 함수 ─────────────────────────────────────────────────────────────

def encode_bgr_to_jpeg_bytes(image: np.ndarray, quality: int = 90) -> bytes:
    """
    BGR ndarray 를 JPEG bytes 로 인코딩합니다.

    Parameters
    ----------
    image   : BGR uint8 ndarray
    quality : JPEG 압축 품질 (1–100)

    Raises
    ------
    RuntimeError : cv2.imencode 실패 시

    Returns
    -------
    bytes : JPEG 바이트
    """
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(
            "cv2.imencode 실패 — 이미지를 JPEG 로 인코딩할 수 없습니다."
        )
    return buf.tobytes()
