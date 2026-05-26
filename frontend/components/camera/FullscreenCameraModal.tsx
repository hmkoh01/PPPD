"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Button } from "@/components/ui/Button";
import { blobToObjectUrl, revokeObjectUrl } from "@/lib/image";
import { EdgeOverlay, type OverlayStatus } from "./EdgeOverlay";
import { CapturePreview } from "./CapturePreview";

type AlignmentStatus = Exclude<OverlayStatus, "idle">;

interface FullscreenCameraModalProps {
  open: boolean;
  title?: string;
  overlayImageUrl?: string;
  onClose: () => void;
  onUse: (blob: Blob) => void;
  onValidateCapture?: (blob: Blob) => Promise<{
    ok: boolean;
    message?: string;
    status?: AlignmentStatus;
  }>;
  onAnalyzeFrame?: (blob: Blob) => Promise<{
    ok: boolean;
    message?: string;
    status?: AlignmentStatus;
  }>;
}

const TEXT = {
  defaultTitle: "\uce74\uba54\ub77c \ucd2c\uc601",
  unsupported: "\uc774 \ube0c\ub77c\uc6b0\uc800\uc5d0\uc11c\ub294 \uce74\uba54\ub77c\ub97c \uc0ac\uc6a9\ud560 \uc218 \uc5c6\uc5b4\uc694. \ud30c\uc77c \uc120\ud0dd\uc744 \uc774\uc6a9\ud574 \uc8fc\uc138\uc694.",
  permissionDenied: "\uce74\uba54\ub77c \uad8c\ud55c\uc774 \uac70\ubd80\ub418\uc5c8\uc5b4\uc694. \ube0c\ub77c\uc6b0\uc800 \uc124\uc815\uc5d0\uc11c \uad8c\ud55c\uc744 \ud5c8\uc6a9\ud574 \uc8fc\uc138\uc694.",
  notFound: "\uc0ac\uc6a9 \uac00\ub2a5\ud55c \uce74\uba54\ub77c\ub97c \ucc3e\uc744 \uc218 \uc5c6\uc5b4\uc694.",
  startFailed: "\uce74\uba54\ub77c\ub97c \uc2dc\uc791\ud560 \uc218 \uc5c6\uc5b4\uc694.",
  close: "\ub2eb\uae30",
  retry: "\ub2e4\uc2dc \uc2dc\ub3c4",
  shutter: "\ucd2c\uc601",
  alignIdle: "기준 사진과 비슷하게 맞춰주세요",
  alignPoor: "기준 사진과 아직 어긋나 있어요",
  alignAlmost: "조금만 더 맞춰주세요",
  alignGood: "거의 맞았어요. 조금만 더 맞춰주세요",
  alignLocked: "정렬 완료. 자동으로 촬영할게요",
  capturing: "촬영합니다",
  retakeHint: "구도가 조금 달라요. 다시 한 번 촬영해 주세요.",
};

const FRAME_ANALYSIS_INTERVAL_MS = 850;
const FRAME_ANALYSIS_LONG_SIDE = 560;
const AUTO_CAPTURE_LOCK_HOLD_MS = 1000;
const AUTO_CAPTURE_COUNTDOWN_SECONDS = 2;
const REAR_CAMERA_KEYWORDS = ["back", "rear", "environment", "후면", "뒤"];
const ULTRA_WIDE_CAMERA_KEYWORDS = ["ultra wide", "ultrawide", "ultra-wide", "0.5", "0,5", "초광각"];

function isRearCameraLabel(label: string): boolean {
  const normalized = label.toLowerCase();
  return REAR_CAMERA_KEYWORDS.some((keyword) => normalized.includes(keyword));
}

function isUltraWideCameraLabel(label: string): boolean {
  const normalized = label.toLowerCase();
  return ULTRA_WIDE_CAMERA_KEYWORDS.some((keyword) => normalized.includes(keyword));
}

function choosePreferredCamera(devices: MediaDeviceInfo[]): MediaDeviceInfo | null {
  const cameras = devices.filter((device) => device.kind === "videoinput" && device.deviceId);
  if (cameras.length === 0) return null;

  const rearCameras = cameras.filter((device) => isRearCameraLabel(device.label));
  const candidates = rearCameras.length > 0 ? rearCameras : cameras;
  return candidates.find((device) => !isUltraWideCameraLabel(device.label)) ?? candidates[0] ?? null;
}

async function getEnvironmentStream(deviceId?: string): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    video: {
      ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: "environment" } }),
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });
}

function toAnalysisBlob(canvas: HTMLCanvasElement): Promise<Blob | null> {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.72);
  });
}

function smoothedStatus(history: AlignmentStatus[]): AlignmentStatus {
  const latest = history.at(-1) ?? "poor";
  if (history.filter((status) => status === "locked").length >= 2) return "locked";
  if (history.filter((status) => status === "good").length >= 2) return "good";
  if (history.filter((status) => status === "poor").length >= 2) return "poor";
  if (history.includes("almost") || history.includes("good") || history.includes("locked")) return "almost";
  return latest;
}

export function FullscreenCameraModal({
  open,
  title = TEXT.defaultTitle,
  overlayImageUrl,
  onClose,
  onUse,
  onValidateCapture,
  onAnalyzeFrame,
}: FullscreenCameraModalProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const analysisCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const capturedBlobRef = useRef<Blob | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const analyzingFrameRef = useRef(false);
  const statusHistoryRef = useRef<AlignmentStatus[]>([]);
  const lockedSinceRef = useRef<number | null>(null);
  const autoCaptureInProgressRef = useRef(false);
  const countdownActiveRef = useRef(false);

  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [overlayStatus, setOverlayStatus] = useState<OverlayStatus>("idle");
  const [liveHint, setLiveHint] = useState(TEXT.alignIdle);
  const [autoCountdown, setAutoCountdown] = useState<number | null>(null);
  const [validationMessage, setValidationMessage] = useState<string | null>(null);
  const [validating, setValidating] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraReady(false);
  }, []);

  const clearPreview = useCallback(() => {
    if (previewUrlRef.current) revokeObjectUrl(previewUrlRef.current);
    previewUrlRef.current = null;
    setPreviewUrl(null);
    capturedBlobRef.current = null;
    setValidationMessage(null);
    setValidating(false);
  }, []);

  const resetLiveGuidance = useCallback(() => {
    analyzingFrameRef.current = false;
    statusHistoryRef.current = [];
    lockedSinceRef.current = null;
    autoCaptureInProgressRef.current = false;
    countdownActiveRef.current = false;
    setOverlayStatus("idle");
    setLiveHint(TEXT.alignIdle);
    setAutoCountdown(null);
  }, []);

  const startCamera = useCallback(async () => {
    stopCamera();
    clearPreview();
    setCameraError(null);
    resetLiveGuidance();

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError(TEXT.unsupported);
      return;
    }

    try {
      let stream = await getEnvironmentStream();

      const devices = await navigator.mediaDevices.enumerateDevices();
      const preferredCamera = choosePreferredCamera(devices);
      const currentTrack = stream.getVideoTracks()[0];
      const currentDeviceId = currentTrack?.getSettings().deviceId;

      if (preferredCamera && preferredCamera.deviceId !== currentDeviceId) {
        stream.getTracks().forEach((track) => track.stop());
        stream = await getEnvironmentStream(preferredCamera.deviceId);
      }

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      const message =
        err instanceof Error
          ? err.name === "NotAllowedError"
            ? TEXT.permissionDenied
            : err.name === "NotFoundError"
            ? TEXT.notFound
            : `\uce74\uba54\ub77c \uc624\ub958: ${err.message}`
          : TEXT.startFailed;
      setCameraError(message);
    }
  }, [clearPreview, resetLiveGuidance, stopCamera]);

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    startCamera();
    return () => {
      document.body.style.overflow = previousOverflow;
      stopCamera();
      if (previewUrlRef.current) revokeObjectUrl(previewUrlRef.current);
      previewUrlRef.current = null;
      capturedBlobRef.current = null;
      resetLiveGuidance();
    };
  }, [open, resetLiveGuidance, startCamera, stopCamera]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !cameraReady || previewUrlRef.current) return;

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return;

    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d")?.drawImage(video, 0, 0, width, height);

    canvas.toBlob(
      (blob) => {
        if (!blob || previewUrlRef.current) return;
        stopCamera();
        if (previewUrlRef.current) revokeObjectUrl(previewUrlRef.current);
        capturedBlobRef.current = blob;
        const url = blobToObjectUrl(blob);
        previewUrlRef.current = url;
        setPreviewUrl(url);
        setAutoCountdown(null);
        setLiveHint(TEXT.capturing);
      },
      "image/jpeg",
      0.92,
    );
  }, [cameraReady, stopCamera]);

  useEffect(() => {
    if (!open || !cameraReady || previewUrl || cameraError || !overlayImageUrl || !onAnalyzeFrame) {
      return;
    }

    let cancelled = false;

    const analyzeCurrentFrame = async () => {
      if (cancelled || analyzingFrameRef.current) return;

      const video = videoRef.current;
      const canvas = analysisCanvasRef.current;
      if (!video || !canvas || !video.videoWidth || !video.videoHeight) return;

      const longSide = Math.max(video.videoWidth, video.videoHeight);
      const scale = Math.min(1, FRAME_ANALYSIS_LONG_SIDE / longSide);
      const width = Math.max(1, Math.round(video.videoWidth * scale));
      const height = Math.max(1, Math.round(video.videoHeight * scale));

      canvas.width = width;
      canvas.height = height;
      canvas.getContext("2d")?.drawImage(video, 0, 0, width, height);

      analyzingFrameRef.current = true;
      try {
        const blob = await toAnalysisBlob(canvas);
        if (!blob || cancelled) return;

        const result = await onAnalyzeFrame(blob);
        if (cancelled) return;

        const nextStatus = result.status ?? (result.ok ? "good" : "poor");
        if (countdownActiveRef.current && nextStatus !== "locked") {
          statusHistoryRef.current = [nextStatus];
          lockedSinceRef.current = null;
          autoCaptureInProgressRef.current = false;
          countdownActiveRef.current = false;
          setAutoCountdown(null);
          setOverlayStatus(nextStatus);
          if (nextStatus === "good") {
            setLiveHint(TEXT.alignGood);
          } else if (nextStatus === "almost") {
            setLiveHint(TEXT.alignAlmost);
          } else {
            setLiveHint(TEXT.alignPoor);
          }
          return;
        }

        statusHistoryRef.current = [...statusHistoryRef.current, nextStatus].slice(-3);

        const displayStatus = smoothedStatus(statusHistoryRef.current);
        const now = Date.now();
        if (displayStatus === "locked") {
          if (!lockedSinceRef.current) lockedSinceRef.current = now;
        } else {
          lockedSinceRef.current = null;
        }

        setOverlayStatus(displayStatus);

        if (displayStatus === "locked") {
          setLiveHint(TEXT.alignLocked);
        } else if (displayStatus === "good") {
          setLiveHint(TEXT.alignGood);
        } else if (displayStatus === "almost") {
          setLiveHint(TEXT.alignAlmost);
        } else {
          setLiveHint(TEXT.alignPoor);
        }
      } catch {
        if (!cancelled) {
          statusHistoryRef.current = [];
          lockedSinceRef.current = null;
          setOverlayStatus("idle");
          setAutoCountdown(null);
          countdownActiveRef.current = false;
          setLiveHint(TEXT.alignIdle);
        }
      } finally {
        analyzingFrameRef.current = false;
      }
    };

    void analyzeCurrentFrame();
    const id = window.setInterval(() => {
      void analyzeCurrentFrame();
    }, FRAME_ANALYSIS_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(id);
      analyzingFrameRef.current = false;
    };
  }, [cameraError, cameraReady, onAnalyzeFrame, open, overlayImageUrl, previewUrl]);

  useEffect(() => {
    if (!open || !cameraReady || previewUrl || cameraError || overlayStatus !== "locked") {
      setAutoCountdown(null);
      autoCaptureInProgressRef.current = false;
      return;
    }

    let countdownId: number | undefined;
    let captureId: number | undefined;
    const lockedAt = lockedSinceRef.current ?? Date.now();
    const holdRemaining = Math.max(AUTO_CAPTURE_LOCK_HOLD_MS - (Date.now() - lockedAt), 0);

    const holdId = window.setTimeout(() => {
      if (autoCaptureInProgressRef.current) return;
      countdownActiveRef.current = true;
      setAutoCountdown(AUTO_CAPTURE_COUNTDOWN_SECONDS);

      let remaining = AUTO_CAPTURE_COUNTDOWN_SECONDS;
      countdownId = window.setInterval(() => {
        remaining -= 1;
        if (remaining > 0) {
          setAutoCountdown(remaining);
          return;
        }

        if (countdownId) window.clearInterval(countdownId);
        setLiveHint(TEXT.capturing);
        setAutoCountdown(null);
        countdownActiveRef.current = false;
        autoCaptureInProgressRef.current = true;
        captureId = window.setTimeout(() => {
          captureFrame();
        }, 180);
      }, 1000);
    }, holdRemaining);

    return () => {
      window.clearTimeout(holdId);
      if (countdownId) window.clearInterval(countdownId);
      if (captureId) window.clearTimeout(captureId);
      if (!previewUrlRef.current) {
        setAutoCountdown(null);
        countdownActiveRef.current = false;
        autoCaptureInProgressRef.current = false;
      }
    };
  }, [cameraError, cameraReady, captureFrame, open, overlayStatus, previewUrl]);

  if (!open || !mounted) return null;

  const handleClose = () => {
    stopCamera();
    clearPreview();
    onClose();
  };

  const handleShutter = () => {
    const requiresAlignment = Boolean(overlayImageUrl && onAnalyzeFrame);
    if (requiresAlignment && overlayStatus !== "locked") return;
    captureFrame();
  };

  const handleRetake = () => {
    clearPreview();
    startCamera();
  };

  const handleUse = () => {
    void handleUseAsync();
  };

  const handleUseAsync = async () => {
    const blob = capturedBlobRef.current;
    if (!blob) return;
    if (onValidateCapture) {
      setValidating(true);
      setValidationMessage(null);
      try {
        const result = await onValidateCapture(blob);
        setOverlayStatus(result.status ?? (result.ok ? "good" : "poor"));
        if (!result.ok) {
          setValidationMessage(result.message || TEXT.retakeHint);
          setValidating(false);
          return;
        }
      } catch {
        setValidationMessage("잠시 후 다시 시도해 주세요.");
        setValidating(false);
        return;
      }
    }
    stopCamera();
    onUse(blob);
    clearPreview();
    onClose();
  };

  const displayedHint = autoCountdown ? `${TEXT.capturing} ${autoCountdown}` : liveHint;
  const requiresAlignment = Boolean(overlayImageUrl && onAnalyzeFrame);
  const canManualCapture = cameraReady && (!requiresAlignment || overlayStatus === "locked");

  return createPortal(
    <div className="fixed left-0 top-0 z-[100] h-[100dvh] w-screen overflow-hidden bg-black text-white">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        onCanPlay={() => setCameraReady(true)}
        className="absolute inset-0 h-full w-full object-contain"
      />

      {overlayImageUrl && cameraReady && !previewUrl && (
        <EdgeOverlay imageUrl={overlayImageUrl} status={overlayStatus} hint={displayedHint} />
      )}

      <header className="pointer-events-none absolute inset-x-0 top-0 z-20 bg-gradient-to-b from-black/70 to-transparent px-4 pb-16 pt-[calc(env(safe-area-inset-top)+1.5rem)]">
        <div className="pointer-events-auto flex items-center justify-between">
          <button
            type="button"
            onClick={handleClose}
            className="rounded-full bg-white/15 px-4 py-2 text-sm font-semibold text-white backdrop-blur transition hover:bg-white/25"
            aria-label={`${TEXT.defaultTitle} ${TEXT.close}`}
          >
            {TEXT.close}
          </button>
          <p className="max-w-[60vw] truncate text-sm font-semibold text-white/95">{title}</p>
          <span className="w-14" />
        </div>
      </header>

      {!cameraReady && !cameraError && !previewUrl && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/40">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/30 border-t-white" />
        </div>
      )}

      {cameraError && !previewUrl && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-black px-6 text-center">
          <p className="max-w-sm rounded-[24px] bg-white/10 px-5 py-4 text-sm leading-relaxed text-white/90 backdrop-blur">
            {cameraError}
          </p>
          <div className="flex gap-3">
            <Button variant="secondary" size="md" onClick={startCamera}>
              {TEXT.retry}
            </Button>
            <Button variant="ghost" size="md" onClick={handleClose}>
              {TEXT.close}
            </Button>
          </div>
        </div>
      )}

      {!previewUrl && !cameraError && !requiresAlignment && (
        <footer className="absolute inset-x-0 bottom-0 z-20 flex justify-center bg-gradient-to-t from-black/80 to-transparent px-4 pb-[calc(env(safe-area-inset-bottom)+3.5rem)] pt-20">
          <button
            type="button"
            onClick={handleShutter}
            disabled={!canManualCapture}
            aria-label={TEXT.shutter}
            className="h-20 w-20 rounded-full border-4 border-white bg-white/20 p-1 shadow-[0_0_0_8px_rgba(255,255,255,0.12)] transition disabled:opacity-50"
          >
            <span className="block h-full w-full rounded-full bg-white" />
          </button>
        </footer>
      )}

      {previewUrl && (
        <CapturePreview
          imageUrl={previewUrl}
          title={title}
          onRetake={handleRetake}
          onUse={handleUse}
          validationMessage={validationMessage}
          validating={validating}
        />
      )}

      <canvas ref={canvasRef} className="hidden" />
      <canvas ref={analysisCanvasRef} className="hidden" />
    </div>,
    document.body,
  );
}
