"use client";

import { useEffect, useState } from "react";

import { ErrorMessage } from "@/components/error-message";
import { PageHeading } from "@/components/page-heading";
import { apiRequest } from "@/lib/api";

type DocumentItem = {
  document_id: string;
  title: string;
  source_type: string;
  visibility: string;
  version: string;
  is_active: boolean;
  content_hash: string;
};

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiRequest<DocumentItem[]>("/documents").then(setDocuments).catch((caught: unknown) => {
      setError(caught instanceof Error ? caught.message : "Unable to load documents");
    });
  }, []);

  return (
    <div className="page">
      <PageHeading eyebrow="Corpus lifecycle" title="Documents" description="Logical documents and their immutable versions remain separately inspectable." />
      <ErrorMessage message={error} />
      <div className="tableWrap">
        <table>
          <thead><tr><th>Document</th><th>Version</th><th>Access</th><th>State</th><th>Hash</th></tr></thead>
          <tbody>
            {documents.map((document) => (
              <tr key={`${document.document_id}-${document.version}`}>
                <td><strong>{document.title}</strong><small>{document.document_id}</small></td>
                <td>{document.version}</td><td>{document.visibility}</td>
                <td><span className={`status ${document.is_active ? "answered" : "muted"}`}>{document.is_active ? "active" : "historical"}</span></td>
                <td><code>{document.content_hash.slice(0, 10)}</code></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

