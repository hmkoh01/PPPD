"""
로컬 파일 시스템 기반 간이 DB 유틸리티.

data/db.json  : 호실 상태 및 메타데이터 저장
data/images/  : 캡처·업로드된 이미지 파일 저장

스키마 (room 객체):
    room_id            : str
    status             : "ready" | "checked_in" | "pending_review" | "approved" | "rejected"
    ref_image_path     : str | None
    initial_image_path : str | None
    final_image_path   : str | None
    issues             : list[IssueDict]
    admin_feedback     : str | None

IssueDict:
    box_coords         : [x, y, w, h]
    status             : "red" | "green" | "orange"
    closeup_image_path : str | None
    vlm_reason         : str | None
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np

# ── 경로 상수 ──────────────────────────────────────────────────────────────
# 이 파일은 utils/ 에 위치하므로, 두 단계 위가 프로젝트 루트입니다.
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
DB_PATH    = DATA_DIR / "db.json"

# 파일 I/O 동시성을 방어하기 위한 전역 Lock
_lock = threading.Lock()


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    """필요한 디렉토리가 없으면 생성합니다."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw() -> dict:
    """db.json을 읽어 파이썬 딕셔너리로 반환합니다. 파일이 없으면 초기 구조를 반환합니다."""
    _ensure_dirs()
    if not DB_PATH.exists():
        return {"rooms": {}}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_raw(db: dict) -> None:
    """딕셔너리를 db.json에 씁니다."""
    _ensure_dirs()
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


# ── 공개 CRUD API ──────────────────────────────────────────────────────────

def get_all_rooms() -> dict[str, dict]:
    """모든 호실 데이터를 {room_id: room_dict} 형태로 반환합니다."""
    with _lock:
        return _load_raw().get("rooms", {})


def get_room(room_id: str) -> dict | None:
    """특정 호실 데이터를 반환합니다. 없으면 None."""
    with _lock:
        return _load_raw()["rooms"].get(room_id)


def create_room(room_id: str) -> dict:
    """
    새 호실을 생성합니다. 이미 존재하면 기존 데이터를 반환합니다.
    초기 status는 "ready"입니다.
    """
    with _lock:
        db = _load_raw()
        if room_id in db["rooms"]:
            return db["rooms"][room_id]
        room: dict = {
            "room_id":            room_id,
            "status":             "ready",
            "ref_image_path":     None,
            "initial_image_path": None,
            "final_image_path":   None,
            "issues":             [],
            "admin_feedback":     None,
        }
        db["rooms"][room_id] = room
        _save_raw(db)
        return room


def update_room(room_id: str, **kwargs) -> dict:
    """
    특정 호실의 필드를 업데이트합니다.
    예: update_room("101", status="approved", admin_feedback="이상 없음")
    """
    with _lock:
        db = _load_raw()
        if room_id not in db["rooms"]:
            raise KeyError(f"호실 '{room_id}'을 찾을 수 없습니다.")
        db["rooms"][room_id].update(kwargs)
        _save_raw(db)
        return db["rooms"][room_id]


def save_image_ndarray(room_id: str, image_type: str, img: np.ndarray) -> str:
    """
    numpy BGR 이미지를 JPEG로 저장하고 절대 경로 문자열을 반환합니다.

    Parameters
    ----------
    room_id    : 호실 ID
    image_type : 파일명 구분자 ("ref", "initial", "final", "closeup_0" 등)
    img        : BGR ndarray

    Returns
    -------
    str : 저장된 파일의 절대 경로
    """
    _ensure_dirs()
    filename = f"{room_id}_{image_type}.jpg"
    path     = IMAGES_DIR / filename
    # cv2.imwrite는 Windows에서 비ASCII 경로를 처리하지 못하므로
    # imencode → bytes → open() 방식으로 저장합니다.
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError(f"이미지 인코딩 실패: {filename}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return str(path)


def save_image_bytes(room_id: str, image_type: str, data: bytes) -> str:
    """
    bytes(JPEG/PNG)를 파일로 저장하고 절대 경로를 반환합니다.
    st.file_uploader / st.camera_input 결과를 바로 저장할 때 사용합니다.
    """
    _ensure_dirs()
    filename = f"{room_id}_{image_type}.jpg"
    path     = IMAGES_DIR / filename
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


def load_image(path: str | None) -> np.ndarray | None:
    """
    절대 경로로 이미지를 BGR ndarray로 읽어 반환합니다.
    경로가 없거나 파일이 없으면 None을 반환합니다.

    cv2.imread는 Windows에서 비ASCII 경로(한글 등)를 처리하지 못하므로
    np.fromfile + cv2.imdecode를 사용합니다.
    """
    if not path or not Path(path).exists():
        return None
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Streamlit st.image()에 전달하기 위해 BGR → RGB 변환합니다."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
