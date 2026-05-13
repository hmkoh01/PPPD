/**
 * 관리자 화면 공통 레이아웃 — 사이드바 + 메인 컨텐츠
 */
"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";

const NAV = [
  { href: "/admin",         label: "대시보드",   icon: "🏠" },
  { href: "/admin/rooms",   label: "호실 관리",  icon: "🏢" },
  { href: "/admin/reviews", label: "점검 리뷰",  icon: "📋" },
];

interface AdminShellProps {
  children: React.ReactNode;
}

export function AdminShell({ children }: AdminShellProps) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* 사이드바 */}
      <aside className="hidden md:flex flex-col w-56 bg-white border-r border-gray-100 py-6 px-4 gap-1">
        <div className="mb-6 px-2">
          <span className="font-bold text-gray-900 text-lg">관리자</span>
          <p className="text-xs text-gray-400 mt-0.5">기숙사 퇴사 점검</p>
        </div>
        {NAV.map((item) => {
          const active = pathname === item.href || (item.href !== "/admin" && pathname.startsWith(item.href));
          return (
            <Link
              key={item.href}
              href={item.href}
              className={[
                "flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors",
                active
                  ? "bg-indigo-50 text-indigo-700"
                  : "text-gray-600 hover:bg-gray-100",
              ].join(" ")}
            >
              <span>{item.icon}</span>
              {item.label}
            </Link>
          );
        })}
      </aside>

      {/* 모바일 하단 탭 */}
      <nav className="md:hidden fixed bottom-0 inset-x-0 bg-white border-t border-gray-200 flex z-10">
        {NAV.map((item) => {
          const active = pathname === item.href || (item.href !== "/admin" && pathname.startsWith(item.href));
          return (
            <Link
              key={item.href}
              href={item.href}
              className={[
                "flex-1 flex flex-col items-center py-2 gap-0.5 text-xs font-medium",
                active ? "text-indigo-700" : "text-gray-400",
              ].join(" ")}
            >
              <span className="text-lg">{item.icon}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* 메인 */}
      <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 py-6 pb-24 md:pb-6">
        {children}
      </main>
    </div>
  );
}
