# Camera 컴포넌트 (Phase 10/11 구현 완료)

## 파일 구조

```
components/camera/
├── CameraCapture.tsx        # 메인 카메라 컴포넌트
├── CapturePreviewModal.tsx  # 촬영 결과 확인 모달
├── EdgeOverlay.tsx          # 구도 가이드 오버레이 (현재 반투명 이미지, Phase 13 Canny TODO)
└── README.md
```

## CameraCapture

```tsx
import { CameraCapture } from "@/components/camera/CameraCapture";

<CameraCapture
  mode="checkin"              // "checkin" | "checkout" | "closeup"
  overlayImageUrl={refUrl}   // 선택: 반투명 가이드 이미지 URL
  onCapture={(blob) => { /* JPEG Blob → API 업로드 */ }}
  onCancel={() => { /* 취소 처리 */ }}
/>
```

| prop | 타입 | 설명 |
|---|---|---|
| `mode` | `CameraMode` | 촬영 목적 (제목/설명 자동 설정) |
| `title` | `string?` | 커스텀 제목 |
| `description` | `string?` | 커스텀 설명 |
| `overlayImageUrl` | `string?` | 반투명 가이드 이미지 URL |
| `onCapture` | `(blob: Blob) => void` | 촬영 완료 콜백 — JPEG Blob |
| `onCancel` | `() => void?` | 취소 버튼 콜백 |

## 실제 사용 패턴

### checkin (`/student/checkin`)

```tsx
// refImageUrl을 오버레이로 사용, 촬영 후 uploadInitialImage 호출
const handleCapture = async (blob: Blob) => {
  setUploading(true);
  const res = await uploadInitialImage(session.inspectionId, blob);
  updateDormitorySession({ status: "checked_in", initialImageUrl: res.initial_image_url });
  router.push("/student/checkout");
};

<CameraCapture mode="checkin" overlayImageUrl={refUrl} onCapture={handleCapture} />
```

### checkout (`/student/checkout`)

```tsx
// initialImageUrl을 오버레이로 사용, 촬영 후 uploadFinalImage → CV 분석
const handleCapture = async (blob: Blob) => {
  setUploading(true);
  const res = await uploadFinalImage(session.inspectionId, blob);
  // res.issues.length > 0 → issues 페이지
  // 없으면 submitInspection → result 페이지
};

<CameraCapture mode="checkout" overlayImageUrl={initUrl} onCapture={handleCapture} />
```

### closeup (`/student/issues`)

```tsx
// crop_image_url을 오버레이로 사용, 촬영 후 uploadIssueCloseup → Gemini 분석
const handleCapture = async (blob: Blob) => {
  setUploading(true);
  const res = await uploadIssueCloseup(issue.id, blob);
  // res.issue.status: "green" | "orange"
  onUploaded(res.issue);
};

<CameraCapture mode="closeup" overlayImageUrl={cropUrl} onCapture={handleCapture} />
```

## 동작 흐름

1. 마운트 시 `getUserMedia` 로 후면 카메라 스트림 획득
2. `<video>` 태그에 스트림 연결, `onCanPlay` 이벤트로 준비 상태 감지
3. `overlayImageUrl` 이 있으면 `EdgeOverlay` 오버레이 표시 (opacity 0.3)
4. 셔터 버튼 → `canvas.toBlob("image/jpeg", 0.92)` → `CapturePreviewModal` 표시
5. "이 사진 사용" → `onCapture(blob)` 호출
6. "다시 촬영" → 모달 닫고 카메라 유지 (스트림 계속)
7. "다시 시도" → 스트림 정리 후 `getUserMedia` 재시도
8. 언마운트 시 `stream.getTracks().forEach(track => track.stop())`

## EdgeOverlay

현재 구현: `overlayImageUrl`을 `opacity-[0.3]` + `pointer-events-none`으로 오버레이.

**Phase 13 TODO**: `overlayImageUrl` 대신 Canny edge 결과를 실시간으로 Canvas에 렌더링.
백엔드에서 edge 이미지를 받아오거나, 클라이언트에서 WebGL/WASM으로 처리.

## 보안 컨텍스트

`navigator.mediaDevices` 는 HTTPS 또는 `localhost` 에서만 존재합니다.
비보안 컨텍스트에서는 "카메라를 사용하려면 HTTPS 또는 localhost 환경이 필요합니다." 메시지를 표시합니다.

## 테스트 페이지

`http://localhost:3000/camera-test` 에서 각 모드를 독립적으로 테스트할 수 있습니다.
