"use client";

interface EdgeOverlayProps {
  imageUrl: string;
}

export function EdgeOverlay({ imageUrl }: EdgeOverlayProps) {
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={imageUrl}
      alt="촬영 기준 오버레이"
      className="pointer-events-none absolute inset-0 h-full w-full select-none object-contain opacity-[0.28]"
    />
  );
}
