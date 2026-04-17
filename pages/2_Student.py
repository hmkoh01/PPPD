"""
학생(점검대상자) 전용 앱.

Step 1 — 입사 초기 사진 촬영:
    관리자가 등록한 기준 사진을 윤곽선(Edge) 오버레이로 띄워 구도를 맞추고 캡처합니다.

Step 2 — 퇴사 최종 사진 촬영:
    Step 1에서 찍은 초기 사진을 오버레이 가이드로 삼아 최종 사진을 캡처합니다.
    캡처 직후 스트림을 종료하고 align_images + detect_difference를 1회 실행합니다.

Step 3 — 차이점 검증 (Human-in-the-loop):
    AI가 찾은 붉은 박스 영역을 갤러리로 보여주고,
    각 항목마다 클로즈업 촬영 → VLM 모의 분석 → 색상 업데이트를 진행합니다.
    모든 이슈가 초록/주황으로 해소되면 최종 제출 버튼이 활성화됩니다.

주의: 카메라 사용을 위해 반드시 HTTPS 또는 localhost 환경에서 실행해야 합니다.
"""
from __future__ import annotations

import random
import sys
import time
import threading
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from utils.db_utils import (
    bgr_to_rgb,
    get_all_rooms,
    load_image,
    save_image_bytes,
    save_image_ndarray,
    update_room,
)
from vision_utils import ImageAlignmentError, align_images, detect_difference

# ── 전역 상수 ──────────────────────────────────────────────────────────────

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
})

_CMP_SIZE     = (128, 128)
_MSE_THRESHOLD = 3000.0

# Edge 오버레이 파라미터
_EDGE_CANNY_LOW  = 40
_EDGE_CANNY_HIGH = 110
_EDGE_COLOR      = (160, 160, 160)   # 회색 (BGR)
_EDGE_ALPHA      = 0.6

# 이슈 박스 색상 (BGR)
_BOX_COLOR = {
    "red":    (0, 0, 255),
    "orange": (0, 165, 255),
    "green":  (0, 200, 0),
}


# ── 통합 VideoProcessor ────────────────────────────────────────────────────

class VideoProcessor(VideoProcessorBase):
    """
    Edge Overlay 프로세서.

    set_reference(ref_bgr) 로 기준 이미지를 주입합니다.

    recv() 에서는 윤곽선 오버레이 합성 + 중앙 십자선 + MSE 구도 판정만 수행합니다.
    (붉은 박스는 그리지 않습니다 — 캡처 후 1회성 detect_difference로 처리)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # 기준 이미지 관련 데이터
        self._ref_bgr: np.ndarray | None        = None
        self._ref_gray_small: np.ndarray | None = None

        # 오버레이 캐시 (카메라 해상도가 바뀔 때만 재계산)
        self._cached_edge_mask: np.ndarray | None = None   # uint8, 1ch letterbox
        self._cached_frame_size: tuple[int, int]  = (0, 0)

        # 최신 프레임 (캡처 버튼에서 읽어갑니다)
        self._latest_frame: np.ndarray | None = None

        # 구도 일치 여부
        self._is_aligned: bool = False

    # ── 외부 인터페이스 ────────────────────────────────────────────────

    def set_reference(self, ref_bgr: np.ndarray) -> None:
        """
        기준 이미지를 주입합니다.
        기준 이미지가 바뀌면 캐시를 초기화합니다.
        """
        gray       = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, _CMP_SIZE, interpolation=cv2.INTER_AREA)

        with self._lock:
            self._ref_bgr             = ref_bgr.copy()
            self._ref_gray_small      = gray_small
            # 기준 이미지 교체 시 캐시 무효화
            self._cached_edge_mask    = None
            self._cached_frame_size   = (0, 0)

    @property
    def is_aligned(self) -> bool:
        with self._lock:
            return self._is_aligned

    @property
    def latest_frame(self) -> np.ndarray | None:
        """메인 스레드에서 캡처 버튼 클릭 시 최신 프레임을 가져갑니다."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    # ── 캐시 계산 (내부) ───────────────────────────────────────────────

    def _rebuild_cache(self, ref_bgr: np.ndarray, w: int, h: int) -> None:
        """
        카메라 해상도(w, h)에 맞게 Edge 오버레이를 사전 계산합니다.
        Letterbox 방식으로 기준 사진의 종횡비를 보존합니다.
        """
        rh, rw = ref_bgr.shape[:2]
        scale  = min(w / rw, h / rh)
        fit_w  = int(rw * scale)
        fit_h  = int(rh * scale)
        x0     = (w - fit_w) // 2
        y0     = (h - fit_h) // 2

        # -- Edge: 1채널 이진 마스크 letterbox --
        gray_ref   = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        blurred    = cv2.GaussianBlur(gray_ref, (13, 13), 0)
        edges      = cv2.Canny(blurred, _EDGE_CANNY_LOW, _EDGE_CANNY_HIGH)
        kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges      = cv2.dilate(edges, kernel, iterations=3)
        scaled_edge = cv2.resize(edges, (fit_w, fit_h), interpolation=cv2.INTER_NEAREST)
        edge_canvas = np.zeros((h, w), dtype=np.uint8)
        edge_canvas[y0:y0 + fit_h, x0:x0 + fit_w] = scaled_edge

        with self._lock:
            self._cached_edge_mask  = edge_canvas
            self._cached_frame_size = (w, h)

    # ── WebRTC recv ────────────────────────────────────────────────────

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        매 프레임:
        1. 윤곽선(Edge) 오버레이 합성
        2. 중앙 십자선
        3. MSE 구도 일치 판정
        """
        try:
            img_orig = frame.to_ndarray(format="bgr24")
            img      = img_orig.copy()
            h, w     = img.shape[:2]

            with self._lock:
                ref_bgr        = self._ref_bgr
                ref_gray_small = self._ref_gray_small
                self._latest_frame = img_orig

            if ref_bgr is not None:
                # 해상도 변경 시 캐시 재계산 (매 프레임 resize를 피함)
                if (w, h) != self._cached_frame_size or self._cached_edge_mask is None:
                    self._rebuild_cache(ref_bgr, w, h)

                # Edge: mask==255 픽셀에만 직접 색 덧씌움 (검은 그림자 방지)
                mask = self._cached_edge_mask
                if mask is not None:
                    edge_px = mask == 255
                    if edge_px.any():
                        cam_vals = img[edge_px].astype(np.float32)
                        color    = np.array(_EDGE_COLOR, dtype=np.float32)
                        blended  = cam_vals * (1 - _EDGE_ALPHA) + color * _EDGE_ALPHA
                        img[edge_px] = np.clip(blended, 0, 255).astype(np.uint8)

            # 중앙 십자선: 구도 정렬 가이드
            cx, cy = w // 2, h // 2
            cv2.line(img, (cx - 20, cy), (cx + 20, cy), (200, 200, 200), 1, cv2.LINE_AA)
            cv2.line(img, (cx, cy - 20), (cx, cy + 20), (200, 200, 200), 1, cv2.LINE_AA)

            # MSE 구도 일치 판정 (오버레이가 없는 원본 프레임으로 계산)
            if ref_gray_small is not None:
                curr_gray  = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
                curr_small = cv2.resize(curr_gray, _CMP_SIZE, interpolation=cv2.INTER_AREA)
                l2         = cv2.norm(curr_small, ref_gray_small, cv2.NORM_L2)
                mse        = (l2 ** 2) / (_CMP_SIZE[0] * _CMP_SIZE[1])
                with self._lock:
                    self._is_aligned = mse < _MSE_THRESHOLD

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception:
            return frame


# ── VLM 모의 분석 ──────────────────────────────────────────────────────────

def mock_vlm_analyze(image: np.ndarray) -> dict:
    """
    VLM(Vision Language Model) API 호출을 모방하는 함수.
    실제 서비스에서는 Claude Vision API 등으로 교체합니다.

    Returns
    -------
    dict
        {"result": "clean"|"suspicious", "reason": str}
    """
    time.sleep(2)   # API 지연 시뮬레이션
    if random.random() < 0.7:
        return {
            "result": "clean",
            "reason": "이상이 감지되지 않았습니다. 원상태로 보입니다.",
        }
    reasons = [
        "쓰레기 또는 이물질이 바닥에 감지되었습니다.",
        "가구 또는 비품 파손이 의심됩니다.",
        "벽면 오염 또는 낙서가 감지되었습니다.",
        "원상복구가 필요한 추가 비치 물품이 발견됩니다.",
    ]
    return {"result": "suspicious", "reason": random.choice(reasons)}


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────

def draw_issues(img: np.ndarray, issues: list[dict]) -> np.ndarray:
    """이슈 목록을 상태 색상에 맞는 박스로 이미지에 그립니다."""
    out = img.copy()
    for i, issue in enumerate(issues):
        x, y, bw, bh = issue["box_coords"]
        color = _BOX_COLOR.get(issue.get("status", "red"), (0, 0, 255))
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 3)
        cv2.putText(
            out, f"#{i + 1}", (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
        )
    return out


def decode_bytes(data: bytes) -> np.ndarray | None:
    """bytes → BGR ndarray 디코딩."""
    if not data:
        return None
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ── 세션 상태 초기화 ───────────────────────────────────────────────────────

def _init_session_state() -> None:
    defaults: dict = {
        "student_step":              0,         # 0=인증, 1=초기촬영, 2=최종촬영, 3=검증/제출
        "student_id_input":          "",        # Step0 입력 학번
        "student_name_input":        "",        # Step0 입력 이름
        "student_room_id":           None,
        "student_ref_bgr":           None,      # 관리자가 등록한 기준 사진 (ndarray)
        "student_initial_bgr":       None,      # Step1 캡처 초기 사진 (ndarray)
        "student_final_aligned_bgr": None,      # Step2 정합 완료 최종 사진 (ndarray)
        "student_issues":            [],        # [{box_coords, status, closeup_bgr, vlm_reason}]
        "webrtc_key_1":              0,         # Step1 webrtc 리셋용 카운터
        "webrtc_key_2":              0,         # Step2 webrtc 리셋용 카운터
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_flow() -> None:
    """처음부터 다시 시작합니다 (Step 0 인증 화면으로 복귀)."""
    keys_to_reset = [
        "student_step", "student_id_input", "student_name_input",
        "student_room_id",
        "student_ref_bgr", "student_initial_bgr", "student_final_aligned_bgr",
        "student_issues",
    ]
    for k in keys_to_reset:
        if k in st.session_state:
            del st.session_state[k]
    # webrtc를 강제 재시작하기 위해 키를 증가시킵니다.
    st.session_state.webrtc_key_1 = st.session_state.get("webrtc_key_1", 0) + 1
    st.session_state.webrtc_key_2 = st.session_state.get("webrtc_key_2", 0) + 1


# ── Step 렌더러 ────────────────────────────────────────────────────────────

def _render_step0() -> None:
    """
    Step 0: 학생 인증.
    학번과 이름을 입력받아 관리자가 배정한 호실을 자동 매칭합니다.
    """
    st.subheader("Step 0  —  🎓 학생 인증")
    st.caption("배정받은 학번과 이름을 입력하면 호실이 자동으로 확인됩니다.")

    sid   = st.text_input("학번", placeholder="2024123456", key="input_sid")
    sname = st.text_input("이름", placeholder="홍길동",     key="input_sname")

    if st.button("🚀 시작하기", use_container_width=True, type="primary"):
        sid_val   = sid.strip()
        sname_val = sname.strip()
        if not sid_val or not sname_val:
            st.error("학번과 이름을 모두 입력해 주세요.")
            return

        rooms = get_all_rooms()
        matched_room_id = None
        for rid, room in rooms.items():
            if (
                room.get("student_id", "").strip()   == sid_val
                and room.get("student_name", "").strip() == sname_val
            ):
                matched_room_id = rid
                break

        if matched_room_id is None:
            st.warning("일치하는 배정 정보가 없습니다. 학번·이름을 확인하거나 관리자에게 문의하세요.")
            return

        st.session_state.student_id_input   = sid_val
        st.session_state.student_name_input = sname_val
        st.session_state.student_room_id    = matched_room_id
        st.session_state.student_step       = 1
        st.rerun()


def _render_step1() -> None:
    """
    Step 1: 입사 — 초기 사진 촬영.
    기준 사진(ref)을 오버레이로 띄워 구도를 맞추고 캡처합니다.
    """
    st.subheader("Step 1 / 3  —  📷 입사 초기 사진 촬영")
    st.caption("기준 사진 오버레이에 카메라를 맞추고 '캡처' 버튼을 눌러 주세요.")

    # ── 호실 확인 (Step 0에서 자동 배정됨) ───────────────────────────
    room_id = st.session_state.student_room_id
    if not room_id:
        st.error("호실 정보가 없습니다. 처음부터 다시 시작해 주세요.")
        return

    rooms = get_all_rooms()
    room  = rooms.get(room_id)
    if room is None or not room.get("ref_image_path"):
        st.warning(f"호실 **{room_id}** 에 기준 사진이 등록되지 않았습니다. 관리자에게 문의하세요.")
        return

    st.info(f"배정 호실: **{room_id}** ({st.session_state.student_name_input})")

    # 기준 사진 로드
    ref_bgr = load_image(room.get("ref_image_path"))
    if ref_bgr is None:
        st.error("기준 사진을 불러올 수 없습니다.")
        return
    st.session_state.student_ref_bgr = ref_bgr

    # ── WebRTC 스트림 ────────────────────────────────────────────────
    ctx = webrtc_streamer(
        key=f"step1_{st.session_state.webrtc_key_1}",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width":      {"ideal": 1280},
                "height":     {"ideal": 720},
                "frameRate":  {"ideal": 30},
                "facingMode": {"ideal": "environment"},
            },
            "audio": False,
        },
        async_processing=True,
    )

    # VideoProcessor에 기준 이미지 주입
    # (매 Streamlit 리렌더링마다 호출되지만, lock 내부에서 빠르게 처리됨)
    if ctx.video_processor is not None:
        ctx.video_processor.set_reference(ref_bgr)

        if ctx.video_processor.is_aligned:
            st.success("✅ 구도 일치 — 지금 캡처해 주세요!")
        else:
            st.info("🎯 오버레이에 카메라를 맞춰 주세요.")

    # ── 캡처 버튼 ────────────────────────────────────────────────────
    if st.button("📸 초기 사진 캡처", use_container_width=True, type="primary"):
        proc = ctx.video_processor if ctx else None
        frame = proc.latest_frame if proc else None
        if frame is None:
            st.error("카메라가 아직 준비되지 않았습니다. 스트림이 시작된 후 다시 눌러 주세요.")
            return

        # 이미지를 디스크에 저장하고 DB를 업데이트합니다.
        path = save_image_ndarray(room_id, "initial", frame)
        update_room(room_id, initial_image_path=path, status="checked_in")

        # 세션에 ndarray도 보관 (Step 2 오버레이 가이드로 사용)
        st.session_state.student_initial_bgr = frame
        st.session_state.student_step        = 2
        # webrtc_key를 증가시켜 Step1 스트림이 더 이상 사용되지 않도록 합니다.
        st.session_state.webrtc_key_1       += 1
        st.rerun()

    # 기준 사진 미리보기
    with st.expander("📌 기준 사진 보기"):
        st.image(bgr_to_rgb(ref_bgr), use_container_width=True)


def _render_step2() -> None:
    """
    Step 2: 퇴사 — 최종 사진 촬영.
    Step 1에서 찍은 초기 사진을 오버레이 가이드로 삼아 최종 사진을 촬영합니다.
    캡처 직후 스트림을 종료하고 align + detect_difference를 1회 실행합니다.
    """
    st.subheader("Step 2 / 3  —  🚪 퇴사 최종 사진 촬영")
    st.caption("입사 때 찍은 초기 사진 오버레이에 맞춰 촬영하세요.")

    room_id     = st.session_state.student_room_id
    initial_bgr = st.session_state.student_initial_bgr

    if initial_bgr is None:
        # 세션에 없으면 DB에서 다시 로드 (새로고침 대비)
        from utils.db_utils import get_room
        room = get_room(room_id)
        if room:
            initial_bgr = load_image(room.get("initial_image_path"))
            st.session_state.student_initial_bgr = initial_bgr

    if initial_bgr is None:
        st.error("초기 사진을 불러올 수 없습니다. Step 1부터 다시 진행해 주세요.")
        if st.button("↩️ Step 1로 돌아가기"):
            st.session_state.student_step = 1
            st.rerun()
        return

    # ── WebRTC 스트림 ────────────────────────────────────────────────
    ctx = webrtc_streamer(
        key=f"step2_{st.session_state.webrtc_key_2}",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width":      {"ideal": 1280},
                "height":     {"ideal": 720},
                "frameRate":  {"ideal": 30},
                "facingMode": {"ideal": "environment"},
            },
            "audio": False,
        },
        async_processing=True,
    )

    if ctx.video_processor is not None:
        # 이번엔 초기 사진이 오버레이 가이드입니다.
        ctx.video_processor.set_reference(initial_bgr)

        if ctx.video_processor.is_aligned:
            st.success("✅ 구도 일치 — 지금 캡처해 주세요!")
        else:
            st.info("🎯 초기 사진 오버레이에 카메라를 맞춰 주세요.")

    # ── 캡처 버튼 ────────────────────────────────────────────────────
    if st.button("📸 최종 사진 캡처", use_container_width=True, type="primary"):
        proc  = ctx.video_processor if ctx else None
        frame = proc.latest_frame if proc else None
        if frame is None:
            st.error("카메라가 아직 준비되지 않았습니다.")
            return

        # webrtc_key 증가 → 다음 rerun에서 Step2 스트림이 렌더링되지 않음 (자동 종료)
        st.session_state.webrtc_key_2 += 1

        # ── 정합 + 차이 검출 (1회, 스피너 표시) ──────────────────────
        with st.spinner("🔍 AI가 차이점을 분석 중입니다…"):
            try:
                aligned = align_images(initial_bgr, frame)
                boxes   = detect_difference(initial_bgr, aligned)
            except ImageAlignmentError as e:
                st.error(f"정합 실패: {e}\n\nStep 1부터 다시 진행해 주세요.")
                return
            except Exception as e:
                st.error(f"분석 오류: {e}")
                return

        # 정합된 최종 이미지와 이슈 목록을 세션에 저장
        st.session_state.student_final_aligned_bgr = aligned
        st.session_state.student_issues = [
            {
                "box_coords":  list(b),
                "status":      "red",     # 초기값: 미검증 (붉은색)
                "closeup_bgr": None,
                "vlm_reason":  None,
            }
            for b in boxes
        ]

        # DB에 최종 사진 저장 (정합된 버전)
        path = save_image_ndarray(room_id, "final", aligned)
        update_room(room_id, final_image_path=path)

        st.session_state.student_step = 3
        st.rerun()

    # 초기 사진 미리보기
    with st.expander("📌 초기 사진 (가이드) 보기"):
        st.image(bgr_to_rgb(initial_bgr), use_container_width=True)


def _render_step3() -> None:
    """
    Step 3: 차이점 검증 (Human-in-the-loop).

    - 최종 사진에 붉은 박스를 표시합니다.
    - 각 박스에 대해 클로즈업 촬영 → VLM 모의 분석 → 색상 업데이트를 진행합니다.
    - 모든 이슈가 초록/주황으로 해소되면 최종 제출 버튼이 활성화됩니다.
    """
    st.subheader("Step 3 / 3  —  🔍 차이점 검증 및 제출")

    room_id     = st.session_state.student_room_id
    final_img   = st.session_state.student_final_aligned_bgr
    issues      = st.session_state.student_issues

    if final_img is None:
        st.error("최종 사진이 없습니다. Step 2부터 다시 진행해 주세요.")
        return

    # ── 상단: 주석 달린 최종 사진 ────────────────────────────────────
    annotated = draw_issues(final_img, issues)
    st.image(bgr_to_rgb(annotated), caption="최종 사진 (박스 색상: 🔴미검증 / 🟢이상없음 / 🟠의심)", use_container_width=True)

    if not issues:
        st.success("AI가 차이를 감지하지 못했습니다. 바로 제출할 수 있습니다.")
    else:
        st.markdown(f"**{len(issues)}곳의 차이 영역이 감지되었습니다.** 각 항목을 클로즈업 촬영해 주세요.")

    st.divider()

    # ── 크롭 갤러리 + 클로즈업 촬영 + VLM 분석 ───────────────────────
    # 이슈를 한 행에 최대 3개씩 갤러리 형태로 배치합니다.
    for i in range(0, len(issues), 3):
        cols = st.columns(min(3, len(issues) - i))
        for j, col in enumerate(cols):
            idx   = i + j
            issue = issues[idx]

            status_emoji = {"red": "🔴", "orange": "🟠", "green": "🟢"}.get(
                issue["status"], "🔴"
            )
            col.markdown(f"**#{idx + 1} {status_emoji}**")

            # 박스 크롭 썸네일
            x, y, bw, bh = issue["box_coords"]
            crop = final_img[y : y + bh, x : x + bw]
            if crop.size > 0:
                col.image(bgr_to_rgb(crop), caption=f"크롭 #{idx + 1}", use_container_width=True)

            # VLM 사유 표시 (이미 분석된 경우)
            if issue["vlm_reason"]:
                col.caption(f"🤖 {issue['vlm_reason']}")
                # 주황(의심) 결과인 경우 — 자율 조치 기회 제공
                if issue["status"] == "orange":
                    col.warning("먼지나 쓰레기가 감지되었습니다. 직접 청소 후 다시 촬영하시겠습니까?")
                    if col.button("🔄 다시 촬영하기", key=f"retake_btn_{idx}"):
                        st.session_state.student_issues[idx]["status"]      = "red"
                        st.session_state.student_issues[idx]["vlm_reason"]  = None
                        st.session_state.student_issues[idx]["closeup_bgr"] = None
                        st.session_state[f"show_closeup_{idx}"] = False
                        st.rerun()
                continue   # 이미 처리된 항목은 아래 버튼 생략

            # 클로즈업 촬영 버튼 → camera_input 열기
            if col.button(f"🔍 클로즈업 촬영", key=f"closeup_btn_{idx}"):
                st.session_state[f"show_closeup_{idx}"] = True

            # st.camera_input은 상태 플래그로 표시 여부를 제어합니다.
            if st.session_state.get(f"show_closeup_{idx}"):
                closeup_file = col.camera_input(
                    f"#{idx + 1} 부위를 가까이서 찍어 주세요",
                    key=f"closeup_cam_{idx}",
                )
                if closeup_file is not None:
                    closeup_bgr = decode_bytes(closeup_file.getvalue())
                    if closeup_bgr is not None:
                        # VLM 분석 실행 (스피너 표시)
                        with col:
                            with st.spinner("🤖 AI 분석 중…"):
                                result = mock_vlm_analyze(closeup_bgr)

                        # 결과에 따라 이슈 상태 업데이트
                        new_status = "green" if result["result"] == "clean" else "orange"
                        st.session_state.student_issues[idx]["status"]      = new_status
                        st.session_state.student_issues[idx]["vlm_reason"]  = result["reason"]
                        st.session_state.student_issues[idx]["closeup_bgr"] = closeup_bgr
                        st.session_state[f"show_closeup_{idx}"] = False
                        st.rerun()

    st.divider()

    # ── 최종 제출 버튼 ────────────────────────────────────────────────
    # 모든 이슈가 초록(green) 또는 주황(orange)으로 해소되어야 활성화됩니다.
    all_resolved = all(
        iss["status"] in ("green", "orange")
        for iss in issues
    ) if issues else True

    if not all_resolved:
        remaining = sum(1 for iss in issues if iss["status"] == "red")
        st.warning(f"아직 클로즈업 촬영이 필요한 항목이 {remaining}건 남아 있습니다.")

    if st.button(
        "✅ 최종 점검 제출",
        disabled=not all_resolved,
        use_container_width=True,
        type="primary",
    ):
        _submit_to_admin(room_id, issues)


def _submit_to_admin(room_id: str, issues: list[dict]) -> None:
    """
    클로즈업 이미지를 디스크에 저장하고,
    DB 이슈 목록을 업데이트한 뒤 status를 "pending_review"로 변경합니다.
    완료 후 Step 4(대기 화면)로 이동합니다.
    """
    db_issues = []
    for i, issue in enumerate(issues):
        closeup_path = None
        closeup_bgr  = issue.get("closeup_bgr")
        if closeup_bgr is not None:
            closeup_path = save_image_ndarray(room_id, f"closeup_{i}", closeup_bgr)

        db_issues.append({
            "box_coords":          issue["box_coords"],
            "status":              issue["status"],
            "closeup_image_path":  closeup_path,
            "vlm_reason":          issue.get("vlm_reason"),
        })

    update_room(room_id, issues=db_issues, status="pending_review")
    st.session_state.student_step = 4
    st.rerun()


def _render_step4() -> None:
    """
    Step 4: 관리자 검토 대기 / 결과 화면.

    - pending_review: 대기 중 안내 + 새로고침 버튼
    - approved:       승인 완료 메시지 → 퇴사 안내
    - rejected:       관리자 피드백 표시 → 재점검(Step 2) 유도
    """
    from utils.db_utils import get_room

    room_id = st.session_state.student_room_id
    room    = get_room(room_id)

    if room is None:
        st.error("호실 정보를 불러올 수 없습니다.")
        return

    status = room.get("status")

    # ── 대기 중 ──────────────────────────────────────────────────────
    if status == "pending_review":
        st.subheader("⏳ 관리자 검토 대기 중")
        st.info(
            "점검 데이터가 성공적으로 제출되었습니다.\n\n"
            "관리자가 사진을 검토하고 있습니다. 승인이 완료될 때까지 이 화면을 유지해 주세요."
        )
        if st.button("🔄 상태 새로고침", use_container_width=True):
            st.rerun()

    # ── 승인 완료 ─────────────────────────────────────────────────────
    elif status == "approved":
        st.subheader("✅ 점검이 승인되었습니다")
        st.success(
            "관리자가 퇴사 점검을 승인했습니다.\n\n"
            "수고하셨습니다!"
        )
        feedback = room.get("admin_feedback", "").strip()
        if feedback:
            st.info(f"관리자 메모: {feedback}")
        st.balloons()
        if st.button("🏠 처음으로 돌아가기", use_container_width=True):
            _reset_flow()
            st.rerun()

    # ── 재점검 요청 ───────────────────────────────────────────────────
    elif status == "rejected":
        st.subheader("🔴 재점검 요청")
        st.warning("관리자로부터 재점검 요청이 왔습니다. 아래 피드백을 확인하고 다시 촬영해 주세요.")
        feedback = room.get("admin_feedback", "").strip()
        if feedback:
            st.error(f"📋 관리자 피드백\n\n{feedback}")
        else:
            st.caption("(관리자 피드백 없음)")
        st.divider()
        if st.button("➡️ 재점검 계속하기", use_container_width=True, type="primary"):
            # Step 2로 돌아가 재촬영
            st.session_state.student_step    = 2
            st.session_state.student_issues  = []
            st.session_state.webrtc_key_2   += 1
            st.rerun()

    # ── 예외 상태 ─────────────────────────────────────────────────────
    else:
        st.info(f"현재 상태: {status}")
        if st.button("🔄 새로고침", use_container_width=True):
            st.rerun()


# ── 메인 ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="학생 점검 앱", page_icon="📱", layout="wide")
    st.markdown(
        "<style>[data-testid='stSidebarNav']{display:none}</style>",
        unsafe_allow_html=True,
    )

    # 보안 컨텍스트 확인 — HTTP(비localhost)에서는 카메라를 사용할 수 없습니다.
    components.html(
        """
        <script>
        if (!window.isSecureContext) {
            const msg = document.createElement('div');
            msg.style.cssText = 'background:#fff3cd;border:1px solid #ffc107;padding:12px 16px;'
                + 'border-radius:6px;font-family:sans-serif;font-size:14px;color:#856404;margin:4px 0;';
            msg.innerHTML = '<b>⚠️ 카메라 사용 불가 — 보안 컨텍스트가 아닙니다.</b><br>'
                + '브라우저는 <code>localhost</code> 또는 <code>HTTPS</code> 환경에서만 카메라를 허용합니다.<br>'
                + '현재 주소창의 IP 주소를 <b>localhost</b>로 바꿔서 접속해 보세요. '
                + '(예: <code>http://localhost:8501</code>)';
            document.body.appendChild(msg);
        }
        </script>
        """,
        height=0,
    )

    _init_session_state()

    # 사이드바: 진행 상황 및 리셋 버튼
    with st.sidebar:
        st.header("📱 학생 점검 앱")
        step = st.session_state.student_step
        _STEP_LABELS = {
            0: "학생 인증 대기",
            1: "1단계: 초기 사진 촬영",
            2: "2단계: 최종 사진 촬영",
            3: "3단계: 차이점 검증",
            4: "검토 대기 중",
        }
        progress_val = min(step / 3, 1.0) if step > 0 else 0.0
        st.progress(progress_val, text=_STEP_LABELS.get(step, f"단계 {step}"))
        st.divider()
        if st.button("↩️ 처음부터 다시 하기", use_container_width=True):
            _reset_flow()
            st.rerun()

    st.title("📱 학생 퇴사 점검")

    step = st.session_state.student_step
    if step == 0:
        _render_step0()
    elif step == 1:
        _render_step1()
    elif step == 2:
        _render_step2()
    elif step == 3:
        _render_step3()
    elif step == 4:
        _render_step4()


if __name__ == "__main__":
    main()
else:
    main()
