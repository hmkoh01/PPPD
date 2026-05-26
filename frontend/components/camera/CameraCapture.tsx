"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/Button";
import { blobToObjectUrl, revokeObjectUrl } from "@/lib/image";
import { FullscreenCameraModal } from "./FullscreenCameraModal";

export type CameraMode = "checkin" | "checkout" | "closeup";

export interface CameraCaptureProps {
  mode: CameraMode;
  title?: string;
  description?: string;
  captureButtonLabel?: string;
  overlayImageUrl?: string;
  onCapture: (blob: Blob) => void;
  onValidateCapture?: (blob: Blob) => Promise<{
    ok: boolean;
    score?: number;
    message?: string;
    status?: "poor" | "almost" | "good" | "locked";
    ssimReady?: boolean;
  }>;
  onAnalyzeFrame?: (blob: Blob) => Promise<{
    ok: boolean;
    score?: number;
    message?: string;
    status?: "poor" | "almost" | "good" | "locked";
    ssimReady?: boolean;
  }>;
  onCancel?: () => void;
  compact?: boolean;
}

const MODE_DEFAULTS: Record<CameraMode, { title: string; description: string }> = {
  checkin: {
    title: "입사 사진 촬영",
    description: "기준 사진과 비슷한 구도로 방 전체가 보이게 촬영해 주세요.",
  },
  checkout: {
    title: "퇴사 사진 촬영",
    description: "입사 때와 같은 구도로 퇴사 전 상태를 촬영해 주세요.",
  },
  closeup: {
    title: "클로즈업 촬영",
    description: "표시된 영역이 잘 보이도록 가까이에서 촬영해 주세요.",
  },
};

export function CameraCapture({
  mode,
  title,
  description,
  captureButtonLabel,
  overlayImageUrl,
  onCapture,
  onValidateCapture,
  onAnalyzeFrame,
  onCancel,
  compact = false,
}: CameraCaptureProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [capturedUrl, setCapturedUrl] = useState<string | null>(null);
  const [cameraSupported, setCameraSupported] = useState(true);
  const [validationError, setValidationError] = useState<string | null>(null);

  const defaults = MODE_DEFAULTS[mode];
  const displayTitle = title ?? defaults.title;
  const displayDescription = description ?? defaults.description;
  const modalOverlayImageUrl = mode === "closeup" ? undefined : overlayImageUrl;

  useEffect(() => {
    setCameraSupported(Boolean(navigator.mediaDevices?.getUserMedia));
  }, []);

  useEffect(() => {
    return () => {
      if (capturedUrl) revokeObjectUrl(capturedUrl);
    };
  }, [capturedUrl]);

  const handleUse = (blob: Blob) => {
    if (capturedUrl) revokeObjectUrl(capturedUrl);
    setValidationError(null);
    setCapturedUrl(blobToObjectUrl(blob));
    onCapture(blob);
  };

  const handleFallbackFile = async (file: File | undefined) => {
    if (!file) return;
    if (onValidateCapture) {
      try {
        const result = await onValidateCapture(file);
        if (!result.ok) {
          setValidationError(result.message || "구도가 조금 달라요. 다시 한 번 촬영해 주세요.");
          if (fileInputRef.current) fileInputRef.current.value = "";
          return;
        }
      } catch {
        setValidationError("잠시 후 다시 시도해 주세요.");
        if (fileInputRef.current) fileInputRef.current.value = "";
        return;
      }
    }
    handleUse(file);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const content = (
    <>
      {!compact && (
        <div>
          <p className="text-base font-bold text-gray-950">{displayTitle}</p>
          <p className="mt-1 text-sm leading-relaxed text-gray-500">{displayDescription}</p>
        </div>
      )}

      {capturedUrl && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-gray-500">선택된 사진</p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={capturedUrl} alt="선택된 촬영 사진" className="h-40 w-full rounded-[20px] object-cover ring-1 ring-gray-100" />
        </div>
      )}

      <div className="flex gap-2">
        {onCancel && (
          <Button variant="ghost" size="md" fullWidth onClick={onCancel}>
            취소
          </Button>
        )}
        <Button type="button" variant="primary" size="lg" fullWidth onClick={() => setOpen(true)}>
          {capturedUrl ? "다시 촬영하기" : captureButtonLabel ?? "카메라로 촬영하기"}
        </Button>
      </div>

      {!cameraSupported && (
        <div className="rounded-2xl bg-amber-50 px-4 py-3 text-xs text-amber-800 ring-1 ring-amber-100">
          이 브라우저에서는 카메라를 직접 열 수 없어요. 파일에서 선택해 주세요.
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(event) => handleFallbackFile(event.target.files?.[0])}
            className="mt-2 block w-full text-xs text-amber-900 file:mr-3 file:rounded-xl file:border-0 file:bg-amber-100 file:px-3 file:py-2 file:text-xs file:font-medium file:text-amber-900"
          />
        </div>
      )}

      {validationError && (
        <div className="rounded-2xl bg-amber-50 px-4 py-3 text-sm text-amber-800 ring-1 ring-amber-100">
          {validationError}
        </div>
      )}

      <FullscreenCameraModal
        open={open}
        title={displayTitle}
        overlayImageUrl={modalOverlayImageUrl}
        onClose={() => setOpen(false)}
        onUse={handleUse}
        onValidateCapture={onValidateCapture}
        onAnalyzeFrame={onAnalyzeFrame}
      />
    </>
  );

  if (compact) return <div className="space-y-2">{content}</div>;

  return <div className="space-y-4 rounded-[24px] bg-white p-5 ring-1 ring-gray-100">{content}</div>;
}
