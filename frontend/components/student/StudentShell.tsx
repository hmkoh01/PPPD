/**
 * 학생 화면 공통 레이아웃 — 모바일 앱 느낌의 단일 컬럼 레이아웃
 */
interface StudentShellProps {
  children: React.ReactNode;
  title?: string;
  back?: { label: string; href: string };
}

export function StudentShell({ children, title, back }: StudentShellProps) {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* 헤더 */}
      <header className="bg-white border-b border-gray-100 sticky top-0 z-10">
        <div className="max-w-lg mx-auto px-4 h-14 flex items-center gap-3">
          {back && (
            <a
              href={back.href}
              className="text-gray-500 hover:text-gray-800 transition-colors text-sm"
            >
              ← {back.label}
            </a>
          )}
          {title && (
            <span className="font-semibold text-gray-900 text-sm flex-1 text-center">
              {title}
            </span>
          )}
          {/* 오른쪽 빈 공간 (균형) */}
          {back && <span className="text-sm opacity-0 pointer-events-none">placeholder</span>}
        </div>
      </header>

      {/* 본문 */}
      <main className="flex-1 max-w-lg mx-auto w-full px-4 py-6 space-y-4">
        {children}
      </main>
    </div>
  );
}
