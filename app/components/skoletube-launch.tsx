import { useState } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { Path } from "../constant";

// Landing page shown inside the SkoleTube embed (iframe). Instead of running the
// chat inside the iframe, it shows a single big button that opens the real,
// full-page SkoleGPT in a new tab. The button keeps the ?name=&prompt= params, so
// the new tab recreates the shared assistant (see jump-to-chat.tsx).
//
// The button only makes sense when we are embedded. If this page is opened directly
// (not in an iframe) there is nothing to "break out of", so we forward straight to
// the assistant instead of showing the button.
//
// It renders as a fixed, opaque overlay so it covers the app chrome (sidebar etc.)
// and looks like a clean launch screen no matter where it is mounted.
export function SkoletubeLaunch() {
  const location = useLocation();
  const name = new URLSearchParams(location.search).get("name")?.trim();

  // Relative href → resolves against this SkoleGPT origin; target="_blank" opens it
  // as a top-level, first-party page, breaking out of the SkoleTube iframe.
  const assistantHref = `${Path.NewChat}${location.search}`;

  // Computed once on (client-side) mount; true when running inside an iframe.
  const [isEmbedded] = useState(() => {
    try {
      return typeof window !== "undefined" ? window.self !== window.top : true;
    } catch {
      return true;
    }
  });

  // Opened directly (not embedded) → no launch button needed; go to the assistant.
  if (!isEmbedded) {
    return <Navigate to={assistantHref} replace />;
  }

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 20,
        padding: 24,
        textAlign: "center",
        background: "var(--white, #ffffff)",
        color: "var(--black, #1a1a1a)",
      }}
    >
      <div style={{ fontSize: 22, fontWeight: 700 }}>
        {name ? `Assistent Navn: ${name}` : "SkoleGPT-assistent"}
      </div>
      <div style={{ maxWidth: 440, lineHeight: 1.5, opacity: 0.75 }}>
        Chatten åbnes på SkoleGPT i en ny fane.
      </div>
      <a
        href={assistantHref}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          display: "inline-block",
          padding: "18px 36px",
          fontSize: 19,
          fontWeight: 700,
          color: "#ffffff",
          background: "#16a34a",
          borderRadius: 14,
          textDecoration: "none",
          boxShadow: "0 4px 14px rgba(0, 0, 0, 0.18)",
        }}
      >
        Åbn i SkoleGPT
      </a>
    </div>
  );
}
