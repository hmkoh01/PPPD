"""
관리자(동장) 전용 대시보드.

기능:
  - 호실 생성 및 기준 사진(Reference Image) 등록
  - 전체 호실 상태 현황판
  - 퇴사 점검 제출 대기(pending_review) 호실 상세 리뷰
  - 최종 합격(approved) / 재점검(rejected) 판정 및 피드백 작성
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit 멀티페이지 환경에서 프로젝트 루트를 sys.path에 추가합니다.
# pages/ 내 파일은 기본적으로 루트를 참조하지만, utils 패키지 임포트를 보장합니다.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import streamlit as st

from utils.db_utils import (
    bgr_to_rgb,
    create_room,
    get_all_rooms,
    get_room,
    load_image,
    save_image_bytes,
    update_room,
)

# ── 상수 ──────────────────────────────────────────────────────────────────

STATUS_LABELS: dict[str, str] = {
    "ready":          "⚪ 등록 대기",
    "checked_in":     "🟡 입사 완료",
    "pending_review": "🔵 점검 제출됨",
    "approved":       "🟢 합격",
    "rejected":       "🔴 재점검 요청",
}

# 이슈 상태별 박스 색상 (BGR)
ISSUE_COLOR: dict[str, tuple[int, int, int]] = {
    "red":    (0, 0, 255),
    "orange": (0, 165, 255),
    "green":  (0, 200, 0),
}


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────

def draw_issues(img: np.ndarray, issues: list[dict]) -> np.ndarray:
    """
    이슈 목록의 박스를 상태에 따른 색으로 이미지에 그립니다.
    admin 리뷰 화면과 학생 결과 화면 모두에서 사용됩니다.
    """
    out = img.copy()
    for i, issue in enumerate(issues):
        x, y, bw, bh = issue["box_coords"]
        color = ISSUE_COLOR.get(issue.get("status", "red"), (0, 0, 255))
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 3)
        # 박스 번호 레이블
        cv2.putText(
            out, f"#{i + 1}", (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
        )
    return out


def decode_uploaded_image(data: bytes) -> np.ndarray | None:
    """st.file_uploader / st.camera_input 의 bytes를 BGR ndarray로 디코딩합니다."""
    if not data:
        return None
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ── 섹션 렌더러 ────────────────────────────────────────────────────────────

def _render_room_management() -> None:
    """
    [탭 1] 호실 관리
    - 새 호실 생성 + 기준 사진 업로드
    - 전체 호실 상태 현황판
    """
    st.subheader("➕ 호실 생성 및 기준 사진 등록")

    with st.form("create_room_form", clear_on_submit=True):
        room_id = st.text_input("호실 번호 (예: 101, A-205)", placeholder="101")
        ref_file = st.file_uploader(
            "기준 사진 업로드",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
        )
        submitted = st.form_submit_button("✅ 호실 등록", use_container_width=True)

    if submitted:
        if not room_id.strip():
            st.error("호실 번호를 입력해 주세요.")
        elif ref_file is None:
            st.error("기준 사진을 업로드해 주세요.")
        else:
            room_id = room_id.strip()
            create_room(room_id)  # 이미 존재하면 무시
            # 기준 사진 저장
            path = save_image_bytes(room_id, "ref", ref_file.getvalue())
            update_room(room_id, ref_image_path=path, status="ready")
            st.success(f"호실 **{room_id}** 이 등록되었습니다.")
            st.rerun()

    st.divider()
    st.subheader("📋 전체 호실 현황")

    rooms = get_all_rooms()
    if not rooms:
        st.info("등록된 호실이 없습니다.")
        return

    # 컬럼 헤더
    hcol = st.columns([1, 2, 1])
    hcol[0].markdown("**호실**")
    hcol[1].markdown("**상태**")
    hcol[2].markdown("**기준 사진**")
    st.divider()

    for rid, room in rooms.items():
        row = st.columns([1, 2, 1])
        row[0].write(rid)
        row[1].write(STATUS_LABELS.get(room["status"], room["status"]))
        ref_img = load_image(room.get("ref_image_path"))
        if ref_img is not None:
            row[2].image(bgr_to_rgb(ref_img), width=80)
        else:
            row[2].caption("없음")


def _render_review_queue() -> None:
    """
    [탭 2] 퇴사 점검 리뷰
    - pending_review 상태인 호실 목록 표시
    - 선택한 호실의 상세 리뷰 및 합격/반려 판정
    """
    rooms = get_all_rooms()
    pending = {rid: r for rid, r in rooms.items() if r["status"] == "pending_review"}

    if not pending:
        st.info("현재 리뷰 대기 중인 호실이 없습니다.")
        return

    st.subheader(f"🔵 리뷰 대기 호실 ({len(pending)}건)")

    selected = st.selectbox(
        "리뷰할 호실을 선택하세요",
        options=list(pending.keys()),
        format_func=lambda x: f"호실 {x}",
    )

    if not selected:
        return

    _render_detail_review(selected)


def _render_detail_review(room_id: str) -> None:
    """
    특정 호실의 상세 점검 리뷰 화면.
    초기 사진, 최종 사진(이슈 박스 포함), 클로즈업 사진, VLM 사유를 나란히 표시합니다.
    """
    room = get_room(room_id)
    if room is None:
        st.error("호실 정보를 불러올 수 없습니다.")
        return

    st.divider()
    st.subheader(f"🔍 호실 {room_id} 상세 리뷰")

    # ── 초기 / 최종 사진 비교 ────────────────────────────────────────
    col_init, col_final = st.columns(2)

    initial_img = load_image(room.get("initial_image_path"))
    final_img   = load_image(room.get("final_image_path"))

    with col_init:
        st.markdown("**📷 입사 초기 사진**")
        if initial_img is not None:
            st.image(bgr_to_rgb(initial_img), use_container_width=True)
        else:
            st.caption("사진 없음")

    with col_final:
        st.markdown("**📷 퇴사 최종 사진 (이슈 박스)**")
        if final_img is not None:
            issues = room.get("issues", [])
            annotated = draw_issues(final_img, issues) if issues else final_img
            st.image(bgr_to_rgb(annotated), use_container_width=True)
        else:
            st.caption("사진 없음")

    # ── 이슈별 상세 정보 ─────────────────────────────────────────────
    issues = room.get("issues", [])
    if issues:
        st.markdown("---")
        st.markdown("**🔎 이슈 상세 (클로즈업 & VLM 판독)**")

        # 이슈를 한 행에 최대 3개씩 갤러리 형태로 표시
        for i in range(0, len(issues), 3):
            cols = st.columns(min(3, len(issues) - i))
            for j, col in enumerate(cols):
                issue = issues[i + j]
                status_emoji = {"red": "🔴", "orange": "🟠", "green": "🟢"}.get(
                    issue.get("status", "red"), "🔴"
                )
                col.markdown(f"**#{i + j + 1} {status_emoji}**")

                # 최종 사진에서 해당 박스 크롭
                if final_img is not None:
                    x, y, bw, bh = issue["box_coords"]
                    crop = final_img[y : y + bh, x : x + bw]
                    if crop.size > 0:
                        col.image(bgr_to_rgb(crop), caption="크롭", use_container_width=True)

                # 클로즈업 사진
                closeup = load_image(issue.get("closeup_image_path"))
                if closeup is not None:
                    col.image(bgr_to_rgb(closeup), caption="클로즈업", use_container_width=True)

                # VLM 판독 사유
                reason = issue.get("vlm_reason")
                if reason:
                    col.caption(f"🤖 {reason}")
    else:
        st.info("감지된 이슈가 없습니다.")

    # ── 최종 판정 ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**✍️ 최종 판정**")

    feedback = st.text_area(
        "관리자 피드백 (선택 사항)",
        value=room.get("admin_feedback") or "",
        key=f"feedback_{room_id}",
        placeholder="추가 소견이나 재점검 요청 사유를 입력하세요.",
    )

    btn_col1, btn_col2 = st.columns(2)
    if btn_col1.button("✅ 합격", key=f"approve_{room_id}", use_container_width=True, type="primary"):
        update_room(room_id, status="approved", admin_feedback=feedback)
        st.success(f"호실 {room_id} — 합격 처리되었습니다.")
        st.rerun()

    if btn_col2.button("❌ 재점검 요청", key=f"reject_{room_id}", use_container_width=True):
        update_room(room_id, status="rejected", admin_feedback=feedback)
        st.warning(f"호실 {room_id} — 재점검 요청이 전달되었습니다.")
        st.rerun()


# ── 메인 ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="관리자 대시보드", page_icon="🔑", layout="wide")
    st.markdown(
        "<style>[data-testid='stSidebarNav']{display:none}</style>",
        unsafe_allow_html=True,
    )
    st.title("🔑 관리자 대시보드")
    st.caption("호실 등록 및 퇴사 점검 결과를 관리합니다.")

    tab_manage, tab_review = st.tabs(["🏠 호실 관리", "📋 점검 리뷰"])

    with tab_manage:
        _render_room_management()

    with tab_review:
        _render_review_queue()


if __name__ == "__main__":
    main()
else:
    # Streamlit 멀티페이지: 파일이 직접 실행되지 않아도 main()을 호출합니다.
    main()
