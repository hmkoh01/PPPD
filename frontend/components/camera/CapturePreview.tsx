"use client";

import { Button } from "@/components/ui/Button";

interface CapturePreviewProps {
  imageUrl: string;
  title?: string;
  onUse: () => void;
  onRetake: () => void;
  validationMessage?: string | null;
  validating?: boolean;
}

const TEXT = {
  title: "\ucd2c\uc601 \uacb0\uacfc",
  retake: "\ub2e4\uc2dc \ucd2c\uc601",
  use: "\uc774 \uc0ac\uc9c4 \uc0ac\uc6a9",
  checking: "확인 중...",
};

export function CapturePreview({
  imageUrl,
  title = TEXT.title,
  onUse,
  onRetake,
  validationMessage,
  validating = false,
}: CapturePreviewProps) {
  return (
    <div className="absolute left-0 top-0 h-[100dvh] w-screen overflow-hidden bg-black">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={imageUrl} alt={title} className="absolute inset-0 h-full w-full bg-black object-contain" />

      <footer className="absolute inset-x-0 bottom-0 z-20 bg-gradient-to-t from-black/85 to-transparent px-4 pb-[calc(env(safe-area-inset-bottom)+3.5rem)] pt-20">
        <div className="mx-auto max-w-sm space-y-3">
          {validationMessage && (
            <p className="rounded-2xl bg-white/12 px-4 py-3 text-center text-sm font-semibold text-white/95 backdrop-blur">
              {validationMessage}
            </p>
          )}
          <div className="flex gap-3">
          <Button variant="secondary" size="lg" fullWidth onClick={onRetake} disabled={validating}>
            {TEXT.retake}
          </Button>
          <Button variant="primary" size="lg" fullWidth onClick={onUse} disabled={validating}>
            {validating ? TEXT.checking : TEXT.use}
          </Button>
          </div>
        </div>
      </footer>
    </div>
  );
}
