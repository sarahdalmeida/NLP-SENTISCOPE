import { models } from "../data";
import type { BackendHealth, DatasetSummary } from "../types";
import { SectionHeading } from "./SectionHeading";

export function DashboardSection({
  health,
  summary,
  healthError,
}: {
  health: BackendHealth | null;
  summary: DatasetSummary | null;
  healthError: string;
}) {
  return (
    <section className="content-section dashboard-section" id="dashboard">
      <SectionHeading title="Dashboard" description="Current dataset and model status." />
      {healthError && <p className="dashboard-error" role="status">{healthError}</p>}
      <div className="dashboard-model-grid">
        {models.map((model) => {
          const status = health?.models.find((item) => item.id === model.id);
          const state = status?.training
            ? "Training"
            : status?.trained
              ? "Trained"
              : status?.training_status === "failed"
                ? "Training failed"
                : health
                  ? "Not trained"
                  : "Unavailable";
          return (
            <article className="dashboard-model-card" key={model.id}>
              <span className={`status-indicator ${status?.trained ? "status-trained" : ""}`} />
              <div><h3>{model.name}</h3><p>{state}</p></div>
            </article>
          );
        })}
      </div>
      {summary ? (
        <div className="dashboard-data">
          <div className="dashboard-data-summary">
            <h3>Dataset</h3>
            <span>{summary.source}</span>
            <strong>{summary.total_records.toLocaleString()} records</strong>
            <p>Train {summary.splits.train.toLocaleString()} · Validation {summary.splits.validation.toLocaleString()} · Test {summary.splits.test.toLocaleString()}</p>
          </div>
          <div className="dashboard-distribution">
            <h3>Sentiment distribution</h3>
            {(["positive", "negative", "neutral"] as const).map((sentiment) => (
              <div className={`dashboard-distribution-row sentiment-${sentiment}`} key={sentiment}>
                <span>{sentiment}</span>
                <strong>{summary.distribution[sentiment].count.toLocaleString()}</strong>
                <span>{(summary.distribution[sentiment].proportion * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          <div className="dashboard-evaluation">
            <h3>Validation evaluation</h3>
            {health?.models.some((item) => item.evaluation) ? (
              <div className="dashboard-evaluation-list">
                {models.map((model) => {
                  const evaluation = health.models.find((item) => item.id === model.id)?.evaluation;
                  return <div key={model.id}><span>{model.shortName}</span><strong>{evaluation ? `Accuracy ${(evaluation.accuracy * 100).toFixed(1)}% · F1 ${(evaluation.f1_macro * 100).toFixed(1)}%` : "Not evaluated"}</strong></div>;
                })}
              </div>
            ) : <p>Evaluation metrics will appear after the models are trained.</p>}
          </div>
        </div>
      ) : (
        <div className="dashboard-empty">
          <h3>Dataset information unavailable</h3>
          <p>{health?.dataset.error || (health?.dataset.status === "loading" ? "TweetEval is loading." : "Connect the backend to view dataset statistics and evaluation results.")}</p>
        </div>
      )}
    </section>
  );
}
