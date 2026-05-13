"use client";

/**
 * EdgeOverlay — 카메라 위에 겹쳐 보이는 반투명 가이드 이미지.
 *
 * Phase 13 TODO: overlayImageUrl 대신 Canny edge 결과를 실시간으로 렌더링.
 */

interface EdgeOverlayProps {
  /** 가이드로 보여줄 이미지 URL (기준 사진 또는 입사 사진) */
  imageUrl: string;
}

export function EdgeOverlay({ imageUrl }: EdgeOverlayProps) {
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={imageUrl}
      alt="구도 가이드"
      className="absolute inset-0 w-full h-full object-contain opacity-[0.3] pointer-events-none select-none"
    />
  );
}
