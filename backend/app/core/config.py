"""
애플리케이션 설정.

pydantic-settings 를 사용하여 환경변수와 .env 파일에서 값을 읽습니다.
모든 설정은 이 모듈의 `settings` 싱글턴 객체를 통해 접근합니다.

사용법:
    from app.core.config import settings

    dir_path = settings.image_storage_path   # Path 객체
    api_key  = settings.GEMINI_API_KEY
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # backend/ 디렉터리 기준으로 .env 파일을 절대경로로 지정
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # .env 에 알 수 없는 키가 있어도 오류 없이 무시
    )

    # ── 데이터베이스 ────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./data/db.sqlite"

    # ── Gemini AI ───────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # ── 이미지 저장 ─────────────────────────────────────────────────────
    # 상대경로는 uvicorn 실행 디렉터리(backend/) 기준
    IMAGE_STORAGE_DIR: str = "./data/images"

    # ── 서버 ────────────────────────────────────────────────────────────
    BACKEND_PORT: int = 8000

    # ── Public URL (이미지 URL 생성에 사용) ─────────────────────────────
    # 개발 환경 기본값; 운영 배포 시 .env 에서 실제 도메인으로 교체
    BASE_URL: str = "http://localhost:8000"

    # ── 파생 속성 ───────────────────────────────────────────────────────

    @property
    def image_storage_path(self) -> Path:
        """IMAGE_STORAGE_DIR 을 Path 객체로 반환합니다."""
        return Path(self.IMAGE_STORAGE_DIR)


# 앱 전역에서 공유하는 싱글턴
settings = Settings()
