"""
DB 초기화 및 이미지 저장 디렉터리 생성.

FastAPI lifespan 이벤트에서 자동 호출되며,
CLI 에서 직접 실행하여 테이블을 수동으로 생성할 수도 있습니다.

CLI 실행:
    cd backend
    python -m app.db.init_db
"""
from __future__ import annotations

import os
from pathlib import Path

from app.db.database import Base, engine
import app.models  # noqa: F401 — Room, Student, Inspection, Issue 를 Base.metadata 에 등록


def init_db() -> None:
    """
    1. IMAGE_STORAGE_DIR (기본: ./data/images) 디렉터리 생성
    2. SQLite / PostgreSQL 테이블 생성 (이미 존재하면 스킵)

    멱등(idempotent) 함수이므로 서버 재시작마다 호출해도 안전합니다.
    """
    images_dir = Path(os.getenv("IMAGE_STORAGE_DIR", "./data/images"))
    images_dir.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=engine)


# ── CLI 직접 실행 지원 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    # `python app/db/init_db.py` 처럼 직접 실행될 때 backend/ 를 sys.path 에 추가
    _backend_dir = str(_Path(__file__).resolve().parent.parent.parent)
    if _backend_dir not in sys.path:
        sys.path.insert(0, _backend_dir)

    print("DB 초기화 시작...")
    init_db()
    print("완료: 테이블 및 data/images/ 디렉터리가 준비되었습니다.")
    print(f"DB 파일: {os.getenv('DATABASE_URL', 'sqlite:///./data/db.sqlite')}")
