"""Inspection 스키마."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.schemas.issue import IssueOut


class InspectionOut(BaseModel):
    id: int
    room_id: int
    student_id: int | None = None
    status: str
    admin_feedback: str | None = None
    submitted_at: datetime | None = None
    reviewed_at: datetime | None = None
    ref_image_path: str | None = None
    ref_image_url: str | None = None
    initial_image_path: str | None = None
    initial_image_url: str | None = None
    final_image_path: str | None = None
    final_image_url: str | None = None
    issues: list[IssueOut] = []


class InspectionStatusOut(BaseModel):
    inspection_id: int
    status: str
    admin_feedback: str | None = None


class FinalImageResponse(BaseModel):
    inspection: InspectionOut
    final_image_url: str | None = None
    issues: list[IssueOut] = []


class InitialImageResponse(BaseModel):
    inspection_id: int
    status: str
    initial_image_path: str
    initial_image_url: str | None = None


class AlignmentCheckResponse(BaseModel):
    ok: bool
    score: float
    status: str
    message: str
    hints: list[str] = []
    good_matches: int | None = None
    inlier_ratio: float | None = None
