"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/Button";

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
  title = "촬영 결과 확인",
  onUse,
  onRetake,
  onClose,
  loading = false,
}: CapturePreviewModalProps) {
  // ESC 키로 닫기
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose?.();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open || !imageUrl) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose?.();
      }}
    >
      <div className="bg-white rounded-2xl w-full max-w-sm overflow-hidden shadow-2xl">
        {/* 헤더 */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
          <h2 className="text-sm font-semibold text-gray-800">{title}</h2>
          {onClose && (
            <button
              className="text-gray-400 hover:text-gray-600 transition-colors"
              onClick={onClose}
              aria-label="닫기"
            >
              ✕
            </button>
          )}
        </div>

        {/* 미리보기 이미지 */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imageUrl}
          alt="촬영 미리보기"
          className="w-full aspect-video object-cover bg-gray-100"
        />

        {/* 버튼 */}
        <div className="p-4 flex gap-3">
          <Button
            variant="secondary"
            size="md"
            fullWidth
            disabled={loading}
            onClick={onRetake}
          >
            다시 촬영
          </Button>
          <Button
            variant="primary"
            size="md"
            fullWidth
            disabled={loading}
            onClick={onUse}
          >
            {loading ? "처리 중…" : "이 사진 사용"}
          </Button>
        </div>
      </div>
    </div>
  );
}
