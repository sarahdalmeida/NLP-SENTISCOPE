import type { BackendHealth, DatasetPreview, DatasetSummary } from "../types";
import { SectionHeading } from "./SectionHeading";

const sentiments = ["positive", "negative", "neutral"] as const;

export function DatasetExplorerSection({
  health,
  summary,
  preview,
  query,
  label,
  loading,
  error,
  onQueryChange,
  onLabelChange,
  onRetry,
}: {
  health: BackendHealth | null;
  summary: DatasetSummary | null;
  preview: DatasetPreview | null;
  query: string;
  label: string;
  loading: boolean;
  error: string;
  onQueryChange: (value: string) => void;
  onLabelChange: (value: string) => void;
  onRetry: () => void;
}) {
  const datasetState = health?.dataset.status ?? "not_loaded";
  return (
    <section className="content-section dataset-section" id="dataset">
      <SectionHeading title="Dataset Explorer" description="Official TweetEval sentiment dataset." />
      {!summary ? (
        <div className="dataset-unavailable">
          <div>
            <h3>{datasetState === "loading" ? "Loading TweetEval" : datasetState === "unavailable" ? "Dataset unavailable" : "Dataset information unavailable"}</h3>
            <p>{error || health?.dataset.error || "Connect to the backend to load dataset details and records."}</p>
          </div>
          {datasetState === "unavailable" && <button className="button button-secondary retry-button" onClick={onRetry}>Retry</button>}
        </div>
      ) : (
        <>
          <div className="dataset-overview-grid">
            <article className="dataset-identity-card">
              <h3>{summary.name}</h3>
              <a href={summary.source_url} target="_blank" rel="noreferrer">{summary.source}</a>
            </article>
            <article className="dataset-total-card"><span>Total records</span><strong>{summary.total_records.toLocaleString()}</strong></article>
            {(["train", "validation", "test"] as const).map((split) => (
              <article className="dataset-total-card" key={split}><span>{split} records</span><strong>{summary.splits[split].toLocaleString()}</strong></article>
            ))}
          </div>
          <div className="dataset-distribution">
            <h3>Sentiment distribution</h3>
            {sentiments.map((sentiment) => {
              const item = summary.distribution[sentiment];
              return (
                <div className={`distribution-row sentiment-${sentiment}`} key={sentiment}>
                  <span>{sentiment}</span>
                  <div className="distribution-track"><span style={{ width: `${item.proportion * 100}%` }} /></div>
                  <strong>{item.count.toLocaleString()} · {(item.proportion * 100).toFixed(1)}%</strong>
                </div>
              );
            })}
          </div>
          <div className="preview-heading">
            <h3>Record preview</h3>
            <div className="preview-controls">
              <label className="preview-search"><span className="sr-only">Search dataset text</span><input value={query} onChange={(event) => onQueryChange(event.target.value)} placeholder="Search text…" /></label>
              <label className="preview-filter"><span className="sr-only">Filter dataset by sentiment</span><select value={label} onChange={(event) => onLabelChange(event.target.value)}><option value="all">All sentiments</option><option value="positive">Positive</option><option value="negative">Negative</option><option value="neutral">Neutral</option></select></label>
            </div>
          </div>
          <div className="dataset-table-wrap">
            {loading ? <div className="table-state">Loading dataset records…</div>
              : error ? <div className="table-state table-error" role="alert">{error}</div>
              : preview?.records.length ? (
                <>
                  <table className="dataset-table">
                    <thead><tr><th>Record</th><th>Text</th><th>Sentiment</th><th>Split</th></tr></thead>
                    <tbody>{preview.records.map((record) => <tr key={record.id}><td className="preview-id">{record.id}</td><td className="preview-text">{record.text}</td><td><span className={`record-label record-label-${record.sentiment}`}>{record.sentiment}</span></td><td className="preview-split">{record.split}</td></tr>)}</tbody>
                  </table>
                  <div className="preview-footer"><span>{preview.total_matches.toLocaleString()} matches · showing {preview.records.length}</span><span>{preview.split} split</span></div>
                </>
              ) : <div className="table-state">{preview ? "No records match this search." : "Dataset preview is unavailable."}</div>}
          </div>
        </>
      )}
    </section>
  );
}
