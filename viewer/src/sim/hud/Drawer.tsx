// Side panel that slides in; unmounted content is never focusable (visibility hidden when closed).
import type { ReactNode } from "react";

export function Drawer({ id, title, open, onClose, side = "right", children }: {
  id: string; title: string; open: boolean; onClose(): void; side?: "left" | "right"; children: ReactNode;
}) {
  return (
    <aside id={id} className="sv-glass sv-side" data-side={side} data-open={open} aria-label={title} aria-hidden={!open}>
      <header>
        <h2>{title}</h2>
        <button className="sv-pill" onClick={onClose} aria-label={`close ${title}`}>Close</button>
      </header>
      {open && children}
    </aside>
  );
}
