"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Button } from "@/components/ui/Button";
import { blobToObjectUrl, revokeObjectUrl } from "@/lib/image";
import { EdgeOverlay } from "./EdgeOverlay";
import { CapturePreview } from "./CapturePreview";

interface FullscreenCameraModalProps {
  open: boolean;
  title?: string;
  overlayImageUrl?: string;
  onClose: () => void;
  onUse: (blob: Blob) => void;
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
};

export function FullscreenCameraModal({
  open,
  title = TEXT.defaultTitle,
  overlayImageUrl,
  onClose,
  onUse,
}: FullscreenCameraModalProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const capturedBlobRef = useRef<Blob | null>(null);
  const previewUrlRef = useRef<string | null>(null);

  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
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
  }, []);

  const startCamera = useCallback(async () => {
    stopCamera();
    clearPreview();
    setCameraError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError(TEXT.unsupported);
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
            ? TEXT.permissionDenied
            : err.name === "NotFoundError"
            ? TEXT.notFound
            : `\uce74\uba54\ub77c \uc624\ub958: ${err.message}`
          : TEXT.startFailed;
      setCameraError(message);
    }
  }, [clearPreview, stopCamera]);

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
    };
  }, [open, startCamera, stopCamera]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open || !mounted) return null;

  const handleClose = () => {
    stopCamera();
    clearPreview();
    onClose();
  };

  const handleShutter = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !cameraReady) return;

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return;

    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d")?.drawImage(video, 0, 0, width, height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        stopCamera();
        if (previewUrlRef.current) revokeObjectUrl(previewUrlRef.current);
        capturedBlobRef.current = blob;
        const url = blobToObjectUrl(blob);
        previewUrlRef.current = url;
        setPreviewUrl(url);
      },
      "image/jpeg",
      0.92,
    );
  };

  const handleRetake = () => {
    clearPreview();
    startCamera();
  };

  const handleUse = () => {
    const blob = capturedBlobRef.current;
    if (!blob) return;
    stopCamera();
    onUse(blob);
    clearPreview();
    onClose();
  };

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

      {overlayImageUrl && cameraReady && !previewUrl && <EdgeOverlay imageUrl={overlayImageUrl} />}

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

      {!previewUrl && !cameraError && (
        <footer className="absolute inset-x-0 bottom-0 z-20 flex justify-center bg-gradient-to-t from-black/80 to-transparent px-4 pb-[calc(env(safe-area-inset-bottom)+3.5rem)] pt-20">
          <button
            type="button"
            onClick={handleShutter}
            disabled={!cameraReady}
            aria-label={TEXT.shutter}
            className="h-20 w-20 rounded-full border-4 border-white bg-white/20 p-1 shadow-[0_0_0_8px_rgba(255,255,255,0.12)] transition disabled:opacity-50"
          >
            <span className="block h-full w-full rounded-full bg-white" />
          </button>
        </footer>
      )}

      {previewUrl && <CapturePreview imageUrl={previewUrl} title={title} onRetake={handleRetake} onUse={handleUse} />}

      <canvas ref={canvasRef} className="hidden" />
    </div>,
    document.body,
  );
}
