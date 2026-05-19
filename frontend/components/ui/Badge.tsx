import type { Tone } from "@/lib/status";

interface BadgeProps {
  tone?: Tone;
  children: React.ReactNode;
}

const toneClasses: Record<Tone, string> = {
  gray:   "bg-gray-100 text-gray-700",
  yellow: "bg-amber-100 text-amber-800",
  blue:   "bg-blue-100 text-blue-800",
  green:  "bg-green-100 text-green-800",
  red:    "bg-red-100 text-red-700",
  orange: "bg-orange-100 text-orange-700",
};

export function Badge({ tone = "gray", children }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${toneClasses[tone]}`}
    >
      {children}
    </span>
  );
}
