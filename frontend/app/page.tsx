import Link from "next/link";
import { Button } from "@/components/ui/Button";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50 flex flex-col items-center justify-center px-4">
      <div className="w-full max-w-sm text-center space-y-8">
        {/* 로고 영역 */}
        <div className="space-y-2">
          <div className="text-6xl">🏢</div>
          <h1 className="text-2xl font-bold text-gray-900">기숙사 퇴사 점검</h1>
          <p className="text-sm text-gray-500">
            AI가 도와주는 스마트 퇴사 점검 플랫폼
          </p>
        </div>

        {/* CTA 버튼 */}
        <div className="space-y-3">
          <Link href="/student" className="block">
            <Button variant="primary" size="lg" fullWidth>
              📱 학생 점검 시작
            </Button>
          </Link>
          <Link href="/admin" className="block">
            <Button variant="secondary" size="lg" fullWidth>
              🔒 관리자 화면
            </Button>
          </Link>
        </div>

        <p className="text-xs text-gray-400">
          학생 화면은 카메라가 있는 스마트폰에서 사용하세요.
        </p>
      </div>
    </div>
  );
}
