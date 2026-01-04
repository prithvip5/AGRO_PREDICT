import React from "react";

export const Card = ({ children, className = "", onClick }) => {
  const interactiveStyles = onClick
    ? "cursor-pointer active:scale-[0.98]"
    : "";

  return (
    <div
      onClick={onClick}
      className={`rounded-card bg-white shadow-soft px-4 py-4 mb-4 ${interactiveStyles} ${className}`}
    >
      {children}
    </div>
  );
};


