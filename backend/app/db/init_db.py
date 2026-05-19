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


def _migrate_sqlite(conn) -> None:
    """
    SQLite 환경에서 schema 변경에 필요한 컬럼을 추가합니다.
    이미 존재하는 컬럼은 무시합니다 (멱등).

    주의: SQLite 는 ALTER TABLE ADD COLUMN 만 지원합니다.
    """
    import sqlite3

    if not isinstance(conn, sqlite3.Connection):
        return  # PostgreSQL 등 다른 DB 는 Alembic 등 정식 마이그레이션 도구를 사용하세요.

    cursor = conn.cursor()

    # issues 테이블 — student_note 컬럼 추가 (v2 스키마)
    cursor.execute("PRAGMA table_info(issues)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    if "student_note" not in existing_cols:
        cursor.execute("ALTER TABLE issues ADD COLUMN student_note TEXT")
        print("  [migrate] issues.student_note 컬럼 추가 완료")

    if "candidate_type" not in existing_cols:
        cursor.execute("ALTER TABLE issues ADD COLUMN candidate_type VARCHAR DEFAULT 'small_damage'")
        print("  [migrate] issues.candidate_type 컬럼 추가 완료")

    conn.commit()


def init_db() -> None:
    """
    1. IMAGE_STORAGE_DIR (기본: ./data/images) 디렉터리 생성
    2. SQLite / PostgreSQL 테이블 생성 (이미 존재하면 스킵)
    3. SQLite 환경에서 누락된 컬럼 자동 추가

    멱등(idempotent) 함수이므로 서버 재시작마다 호출해도 안전합니다.
    """
    images_dir = Path(os.getenv("IMAGE_STORAGE_DIR", "./data/images"))
    images_dir.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=engine)

    # SQLite 환경에서 기존 DB 스키마 마이그레이션 (새 컬럼 추가)
    db_url = os.getenv("DATABASE_URL", "sqlite:///./data/db.sqlite")
    if db_url.startswith("sqlite"):
        try:
            import sqlite3
            db_path = db_url.replace("sqlite:///", "")
            with sqlite3.connect(db_path) as conn:
                _migrate_sqlite(conn)
        except Exception as exc:
            print(f"  [migrate] SQLite 마이그레이션 중 오류 (무시됨): {exc}")


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
