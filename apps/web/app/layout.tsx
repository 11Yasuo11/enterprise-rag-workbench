import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "Enterprise RAG Workbench",
  description: "Enterprise-oriented RAG evaluation and retrieval workbench",
};

const links = [
  ["Chat", "/chat"],
  ["Documents", "/documents"],
  ["Retrieval Debug", "/retrieval"],
  ["Evaluations", "/evaluations"],
] as const;

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <aside className="sidebar">
            <Link className="brand" href="/">
              <span className="brandMark">ER</span>
              <span>RAG Workbench</span>
            </Link>
            <nav aria-label="Primary navigation">
              {links.map(([label, href]) => (
                <Link key={href} href={href}>{label}</Link>
              ))}
            </nav>
            <p className="sidebarMeta">AcmeAI synthetic corpus<br />Frozen architecture v2</p>
          </aside>
          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}
