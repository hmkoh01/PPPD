"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/Button";
import { EdgeOverlay } from "./EdgeOverlay";
import { CapturePreviewModal } from "./CapturePreviewModal";
import { blobToObjectUrl, revokeObjectUrl } from "@/lib/image";

export type CameraMode = "checkin" | "checkout" | "closeup";

export interface CameraCaptureProps {
  mode: CameraMode;
  title?: string;
  description?: string;
  overlayImageUrl?: string;
  onCapture: (blob: Blob) => void;
  onCancel?: () => void;
}

const MODE_DEFAULTS: Record<CameraMode, { title: string; description: string }> = {
  checkin: {
    title: "입사 사진 촬영",
    description: "기준 사진의 구도에 맞춰 방 전체를 촬영해 주세요.",
  },
  checkout: {
    title: "퇴사 사진 촬영",
    description: "입사 때와 동일한 각도에서 현재 방 상태를 촬영해 주세요.",
  },
  closeup: {
    title: "클로즈업 촬영",
    description: "표시된 영역을 가까이서 촬영해 주세요.",
  },
};

export function CameraCapture({
  mode,
  title,
  description,
  overlayImageUrl,
  onCapture,
  onCancel,
}: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraReady, setCameraReady] = useState(false);

  // 미리보기 모달 상태
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  const defaults = MODE_DEFAULTS[mode];
  const displayTitle = title ?? defaults.title;
  const displayDescription = description ?? defaults.description;

  // 카메라 시작
  const startCamera = useCallback(async () => {
    // 이전 스트림 정리
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    setCameraError(null);
    setCameraReady(false);

    // 보안 컨텍스트 확인 (HTTPS 또는 localhost 필요)
    if (!navigator.mediaDevices) {
      setCameraError(
        "카메라를 사용하려면 HTTPS 또는 localhost 환경이 필요합니다."
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      const message =
        err instanceof Error
          ? err.name === "NotAllowedError"
            ? "카메라 접근 권한이 없습니다. 브라우저 설정에서 허용해 주세요."
            : err.name === "NotFoundError"
            ? "카메라를 찾을 수 없습니다."
            : err.name === "OverconstrainedError"
            ? "카메라 설정을 지원하지 않습니다. 다시 시도해 주세요."
            : `카메라 오류: ${err.message}`
          : "카메라를 시작할 수 없습니다.";
      setCameraError(message);
    }
  }, []);

  // 스트림 정리
  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, [startCamera, stopCamera]);

  // 이전 previewUrl 해제
  useEffect(() => {
    return () => {
      if (previewUrl) revokeObjectUrl(previewUrl);
    };
  }, [previewUrl]);

  const handleShutter = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !cameraReady) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        if (previewUrl) revokeObjectUrl(previewUrl);
        const url = blobToObjectUrl(blob);
        setCapturedBlob(blob);
        setPreviewUrl(url);
        setModalOpen(true);
      },
      "image/jpeg",
      0.92
    );
  };

  const handleUse = () => {
    if (!capturedBlob) return;
    setModalOpen(false);
    onCapture(capturedBlob);
  };

  const handleRetake = () => {
    setModalOpen(false);
    if (previewUrl) {
      revokeObjectUrl(previewUrl);
      setPreviewUrl(null);
    }
    setCapturedBlob(null);
  };

  const handleModalClose = () => {
    setModalOpen(false);
  };

  return (
    <div className="space-y-3">
      {/* 제목 + 설명 */}
      <div>
        <p className="text-sm font-semibold text-gray-800">{displayTitle}</p>
        <p className="text-xs text-gray-500 mt-0.5">{displayDescription}</p>
      </div>

      {/* 카메라 뷰파인더 */}
      <div className="relative rounded-2xl overflow-hidden bg-gray-900 aspect-video">
        {/* 비디오 */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          onCanPlay={() => setCameraReady(true)}
          className="w-full h-full object-cover"
        />

        {/* 엣지 오버레이 */}
        {overlayImageUrl && cameraReady && (
          <EdgeOverlay imageUrl={overlayImageUrl} />
        )}

        {/* 카메라 오류 */}
        {cameraError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-4 text-center bg-gray-900">
            <span className="text-3xl">📷</span>
            <p className="text-sm text-white/70">{cameraError}</p>
            <Button variant="secondary" size="sm" onClick={startCamera}>
              다시 시도
            </Button>
          </div>
        )}

        {/* 로딩 (카메라 준비 전) */}
        {!cameraError && !cameraReady && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-6 h-6 border-2 border-white/40 border-t-white rounded-full animate-spin" />
          </div>
        )}
      </div>

      {/* 숨겨진 캔버스 */}
      <canvas ref={canvasRef} className="hidden" />

      {/* 버튼 */}
      <div className="flex gap-2">
        {onCancel && (
          <Button variant="ghost" size="md" fullWidth onClick={onCancel}>
            취소
          </Button>
        )}
        <Button
          variant="primary"
          size="lg"
          fullWidth
          disabled={!cameraReady || !!cameraError}
          onClick={handleShutter}
        >
          촬영하기
        </Button>
      </div>

      {/* 미리보기 모달 */}
      <CapturePreviewModal
        open={modalOpen}
        imageUrl={previewUrl}
        title={displayTitle}
        onUse={handleUse}
        onRetake={handleRetake}
        onClose={handleModalClose}
      />
    </div>
  );
}
