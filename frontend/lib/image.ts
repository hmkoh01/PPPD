/**
 * 이미지 URL 헬퍼.
 * backend의 public URL은 NEXT_PUBLIC_API_BASE_URL/images/{filename} 형식입니다.
 */

const configuredBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "";

function isLocalhostUrl(url: string): boolean {
  return /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(url);
}

function shouldUseSameOrigin(baseUrl: string): boolean {
  if (!baseUrl) return true;
  if (typeof window === "undefined") return false;
  if (!isLocalhostUrl(baseUrl)) return false;

  const host = window.location.hostname;
  return host !== "localhost" && host !== "127.0.0.1";
}

function imageBaseUrl(): string {
  return shouldUseSameOrigin(configuredBaseUrl) ? "" : configuredBaseUrl;
}

/**
 * 파일명 또는 절대 URL이 들어오면 표시용 URL을 반환합니다.
 * null/undefined 면 null을 반환합니다.
 */
export function resolveImageUrl(pathOrUrl: string | null | undefined): string | null {
  if (!pathOrUrl) return null;
  // 이미 http로 시작하는 경우 그대로 반환
  if (pathOrUrl.startsWith("http")) {
    try {
      const url = new URL(pathOrUrl);
      if (
        shouldUseSameOrigin(`${url.protocol}//${url.host}`) &&
        url.pathname.startsWith("/images/")
      ) {
        return url.pathname;
      }
    } catch {
      return pathOrUrl;
    }
    return pathOrUrl;
  }
  // 파일명만 온 경우
  const filename = pathOrUrl.split("/").pop() ?? pathOrUrl;
  return `${imageBaseUrl()}/images/${filename}`;
}

/** Blob → object URL. 사용 후 반드시 revokeObjectUrl로 해제하세요. */
export function blobToObjectUrl(blob: Blob): string {
  return URL.createObjectURL(blob);
}

/** createObjectURL로 만든 URL을 해제합니다. */
export function revokeObjectUrl(url: string): void {
  URL.revokeObjectURL(url);
}

/** 브라우저가 표시할 수 있는 이미지 파일인지 확인합니다. */
export function isImageFile(file: File): boolean {
  return file.type.startsWith("image/");
}
