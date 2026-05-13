"""
SQLAlchemy 엔진, 세션, Base 클래스를 정의합니다.
모든 ORM 모델은 이 모듈의 Base를 상속받아야 합니다.

TimestampMixin: created_at / updated_at 자동 관리를 위한 믹스인.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# backend/ 디렉터리 기준으로 .env 로드
load_dotenv(Path(__file__).parent.parent.parent / ".env")

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite:///./data/db.sqlite",
)

# SQLite 전용: 멀티스레드 접근 허용 (FastAPI async 환경 대응)
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """모든 ORM 모델의 베이스 클래스."""
    pass


class TimestampMixin:
    """
    created_at / updated_at 자동 관리 믹스인.

    사용법:
        class MyModel(TimestampMixin, Base):
            ...

    - created_at : INSERT 시 자동으로 현재 시각(UTC) 설정
    - updated_at : INSERT 시 설정, UPDATE 시 자동으로 현재 시각(UTC) 갱신
    """
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        comment="레코드 생성 시각 (UTC)",
    )
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        comment="레코드 최종 수정 시각 (UTC)",
    )


# ── FastAPI Dependency ─────────────────────────────────────────────────────

def get_db():
    """
    요청마다 DB 세션을 생성하고 응답 완료 후 닫습니다.

    Usage:
        @router.get("/rooms")
        def list_rooms(db: Session = Depends(get_db)):
            ...
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
