import type { HTMLAttributes } from "react";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  title?: string;
  description?: string;
}

export function Card({ title, description, className = "", children, ...rest }: CardProps) {
  return (
    <div
      className={`bg-white rounded-2xl shadow-sm border border-gray-100 p-5 ${className}`}
      {...rest}
    >
      {(title || description) && (
        <div className="mb-4">
          {title && <h3 className="text-base font-semibold text-gray-900">{title}</h3>}
          {description && <p className="mt-0.5 text-sm text-gray-500">{description}</p>}
        </div>
      )}
      {children}
    </div>
  );
}
