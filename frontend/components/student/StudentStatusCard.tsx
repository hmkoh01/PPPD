import type { RoomStatus } from "@/lib/types";
import { getStatusLabel, getStatusTone } from "@/lib/status";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";

interface StudentStatusCardProps {
  status: RoomStatus;
  adminFeedback?: string | null;
  children?: React.ReactNode;
}

const messages: Record<RoomStatus, string> = {
  ready: "학생 인증이 완료되었어요. 입사 기준 사진을 촬영해 주세요.",
  checked_in: "입사 사진이 저장되었어요. 퇴사 후 상태를 같은 구도로 촬영해 주세요.",
  pending_review: "제출이 완료되었어요. 관리자가 점검 결과를 확인하고 있습니다.",
  approved: "퇴사 점검이 승인되었어요. 필요한 절차가 완료되었습니다.",
  rejected: "재점검이 필요합니다. 관리자 피드백을 확인하고 다시 제출해 주세요.",
};

const surfaceClasses: Record<RoomStatus, string> = {
  ready: "bg-white",
  checked_in: "bg-blue-50 ring-blue-100",
  pending_review: "bg-amber-50 ring-amber-100",
  approved: "bg-emerald-50 ring-emerald-100",
  rejected: "bg-red-50 ring-red-100",
};

export function StudentStatusCard({
  status,
  adminFeedback,
  children,
}: StudentStatusCardProps) {
  return (
    <Card className={surfaceClasses[status]}>
      <div className="space-y-5">
        <div className="space-y-3">
          <Badge tone={getStatusTone(status)}>{getStatusLabel(status)}</Badge>
          <div>
            <p className="break-keep text-3xl font-extrabold leading-snug tracking-tight text-gray-950">
              {getStatusLabel(status)}
            </p>
            <p className="mt-2 break-keep text-sm leading-snug text-gray-600">{messages[status]}</p>
          </div>
        </div>

        {status === "rejected" && adminFeedback && (
          <div className="rounded-2xl bg-white/80 p-4 text-left ring-1 ring-red-100">
            <p className="mb-1 text-xs font-bold text-red-600">관리자 피드백</p>
            <p className="text-sm leading-relaxed text-red-900">{adminFeedback}</p>
          </div>
        )}

        {children && <div className="space-y-2 pt-1">{children}</div>}
      </div>
    </Card>
  );
}
