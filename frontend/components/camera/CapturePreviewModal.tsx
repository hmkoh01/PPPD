"use client";

import { useEffect } from "react";
import { CapturePreview } from "./CapturePreview";

export interface CapturePreviewModalProps {
  open: boolean;
  imageUrl: string | null;
  title?: string;
  onUse: () => void;
  onRetake: () => void;
  onClose?: () => void;
  loading?: boolean;
}

export function CapturePreviewModal({
  open,
  imageUrl,
  title = "촬영 결과",
  onUse,
  onRetake,
  onClose,
}: CapturePreviewModalProps) {
  useEffect(() => {
    if (!open) return;
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose?.();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open || !imageUrl) return null;

  return (
    <div className="fixed inset-0 z-[100] bg-black">
      {onClose && (
        <button
          type="button"
          onClick={onClose}
          className="absolute left-4 top-4 z-10 rounded-full bg-black/50 px-4 py-2 text-sm font-medium text-white backdrop-blur"
        >
          닫기
        </button>
      )}
      <CapturePreview
        imageUrl={imageUrl}
        title={title}
        onRetake={onRetake}
        onUse={onUse}
      />
    </div>
  );
}
