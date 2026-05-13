import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "기숙사 퇴사 점검",
  description: "기숙사 퇴사 점검 플랫폼",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
