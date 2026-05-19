import type { Room } from "@/lib/types";
import { Card } from "@/components/ui/Card";

interface AdminKpiCardsProps {
  rooms: Room[];
}

export function AdminKpiCards({ rooms }: AdminKpiCardsProps) {
  const total       = rooms.length;
  const checkedIn   = rooms.filter((r) => r.status === "checked_in").length;
  const pending     = rooms.filter((r) => r.status === "pending_review").length;
  const approved    = rooms.filter((r) => r.status === "approved").length;
  const rejected    = rooms.filter((r) => r.status === "rejected").length;

  const kpis = [
    { label: "전체 호실", value: total, color: "text-gray-900", surface: "bg-white" },
    { label: "입사 완료", value: checkedIn, color: "text-blue-600", surface: "bg-blue-50" },
    { label: "검토 대기", value: pending, color: "text-amber-600", surface: "bg-amber-50" },
    { label: "승인 완료", value: approved, color: "text-emerald-600", surface: "bg-emerald-50" },
    { label: "재점검 요청", value: rejected, color: "text-red-600", surface: "bg-red-50" },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
      {kpis.map((k) => (
        <Card key={k.label} className={`!p-4 ${k.surface}`}>
          <div className="flex flex-col gap-1.5">
            <span className={`text-3xl font-extrabold ${k.color}`}>{k.value}</span>
            <span className="text-xs font-semibold leading-tight text-gray-500">{k.label}</span>
          </div>
        </Card>
      ))}
    </div>
  );
}
