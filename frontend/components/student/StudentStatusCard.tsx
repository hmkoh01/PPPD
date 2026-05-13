import type { RoomStatus } from "@/lib/types";
import { getStatusLabel, getStatusTone } from "@/lib/status";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";

interface StudentStatusCardProps {
  status: RoomStatus;
  adminFeedback?: string | null;
  /** 상태별 CTA 버튼 등을 받아 카드 하단에 렌더링합니다. */
  children?: React.ReactNode;
}

const icons: Record<RoomStatus, string> = {
  ready:          "📋",
  checked_in:     "📸",
  pending_review: "⏳",
  approved:       "✅",
  rejected:       "🔴",
};

const messages: Record<RoomStatus, string> = {
  ready:          "관리자가 등록을 완료했습니다. 점검을 시작할 수 있습니다.",
  checked_in:     "입사 사진 촬영이 완료되었습니다. 퇴사 사진을 촬영해 주세요.",
  pending_review: "점검 데이터가 제출되었습니다. 관리자 검토를 기다리고 있습니다.",
  approved:       "관리자가 점검을 승인했습니다. 퇴사 처리가 완료되었습니다.",
  rejected:       "관리자가 재점검을 요청했습니다. 아래 피드백을 확인해 주세요.",
};

export function StudentStatusCard({
  status,
  adminFeedback,
  children,
}: StudentStatusCardProps) {
  return (
    <Card>
      <div className="flex flex-col items-center text-center gap-3 py-2">
        <span className="text-4xl">{icons[status]}</span>
        <Badge tone={getStatusTone(status)}>{getStatusLabel(status)}</Badge>
        <p className="text-sm text-gray-600 leading-relaxed">{messages[status]}</p>

        {status === "rejected" && adminFeedback && (
          <div className="w-full mt-1 bg-red-50 border border-red-200 rounded-xl p-3 text-left">
            <p className="text-xs font-semibold text-red-700 mb-1">관리자 피드백</p>
            <p className="text-sm text-red-800 leading-relaxed">{adminFeedback}</p>
          </div>
        )}

        {children && <div className="w-full mt-1 space-y-2">{children}</div>}
      </div>
    </Card>
  );
}
