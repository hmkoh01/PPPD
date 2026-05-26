"use client";

export type OverlayStatus = "idle" | "poor" | "almost" | "good" | "locked";

interface EdgeOverlayProps {
  imageUrl: string;
  edgeImageUrl?: string;
  status?: OverlayStatus;
  hint?: string;
}

const statusClasses: Record<OverlayStatus, string> = {
  idle: "border-white/20",
  poor: "border-amber-100/30",
  almost: "border-amber-100/35",
  good: "border-emerald-100/40",
  locked: "border-emerald-100/55",
};

export function EdgeOverlay({ imageUrl, edgeImageUrl, status = "idle", hint }: EdgeOverlayProps) {
  return (
    <div className="pointer-events-none absolute inset-0 select-none">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={edgeImageUrl ?? imageUrl}
        alt="촬영 기준 오버레이"
        className="absolute inset-0 h-full w-full object-contain opacity-[0.34] contrast-125"
      />
      <div className="absolute inset-7">
        <div className={`absolute left-0 top-0 h-6 w-6 rounded-tl-xl border-l border-t ${statusClasses[status]}`} />
        <div className={`absolute right-0 top-0 h-6 w-6 rounded-tr-xl border-r border-t ${statusClasses[status]}`} />
        <div className={`absolute bottom-0 left-0 h-6 w-6 rounded-bl-xl border-b border-l ${statusClasses[status]}`} />
        <div className={`absolute bottom-0 right-0 h-6 w-6 rounded-br-xl border-b border-r ${statusClasses[status]}`} />
      </div>
      {hint && (
        <div className="absolute inset-x-4 bottom-[calc(env(safe-area-inset-bottom)+8.25rem)] flex justify-center">
          <p className="rounded-full bg-black/45 px-4 py-2 text-sm font-semibold text-white/95 backdrop-blur">
            {hint}
          </p>
        </div>
      )}
    </div>
  );
}
