"""
기숙사 퇴사 점검 플랫폼 — FastAPI 백엔드 엔트리포인트.

실행:
    cd backend
    uvicorn app.main:app --reload --port 8000

API 문서 (자동 생성):
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.db.init_db import init_db
from app.api import admin, inspections, issues, rooms, students


# ── Lifespan (startup / shutdown) ────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작 시:
      - data/images/ 디렉터리 생성
      - SQLite 테이블 생성 (존재하면 스킵)
    """
    init_db()
    yield


# ── FastAPI 앱 ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="기숙사 퇴사 점검 API",
    description="기숙사 퇴사 점검 플랫폼 백엔드. Next.js 프론트엔드와 통신합니다.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Next.js 개발 서버
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static Files (이미지 서빙) ─────────────────────────────────────────────
# /images/{filename} 경로로 저장된 이미지를 직접 서빙합니다.
# API 라우터(/api/*)와 경로가 겹치지 않으므로 라우터 등록 전에 마운트해도 안전합니다.
# 주의: StaticFiles 는 마운트 시점에 디렉터리가 반드시 존재해야 합니다.

_images_dir = settings.image_storage_path
_images_dir.mkdir(parents=True, exist_ok=True)

app.mount(
    "/images",
    StaticFiles(directory=str(_images_dir)),
    name="images",
)


# ── API 라우터 ─────────────────────────────────────────────────────────────

app.include_router(students.router,    prefix="/api/students",    tags=["students"])
app.include_router(rooms.router,       prefix="/api/rooms",       tags=["rooms"])
app.include_router(inspections.router, prefix="/api/inspections", tags=["inspections"])
app.include_router(issues.router,      prefix="/api/issues",      tags=["issues"])
app.include_router(admin.router,       prefix="/api/admin",       tags=["admin"])


# ── Health Check ──────────────────────────────────────────────────────────

@app.get("/health", tags=["health"], summary="서버 헬스 체크")
def health_check():
    """서버가 정상 작동 중이면 {"status": "ok"}를 반환합니다."""
    return {"status": "ok"}
