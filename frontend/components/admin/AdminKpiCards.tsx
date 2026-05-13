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
    { label: "전체 호실",     value: total,      icon: "🏢", color: "text-gray-700" },
    { label: "입사 완료",     value: checkedIn,  icon: "📷", color: "text-blue-600" },
    { label: "검토 대기",     value: pending,    icon: "⏳", color: "text-yellow-600" },
    { label: "승인 완료",     value: approved,   icon: "✅", color: "text-green-600" },
    { label: "재점검 요청",   value: rejected,   icon: "🔴", color: "text-red-600" },
  ];

  return (
    <div className="grid grid-cols-3 lg:grid-cols-5 gap-3">
      {kpis.map((k) => (
        <Card key={k.label} className="!p-4">
          <div className="flex flex-col gap-1">
            <span className="text-xl">{k.icon}</span>
            <span className={`text-2xl font-bold ${k.color}`}>{k.value}</span>
            <span className="text-xs text-gray-500 leading-tight">{k.label}</span>
          </div>
        </Card>
      ))}
    </div>
  );
}
