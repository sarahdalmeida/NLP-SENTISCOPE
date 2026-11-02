import { Fragment } from "react";
import { models } from "../data";
import type { ModelEvaluation, ModelStatus, Sentiment, SplitEvaluation } from "../types";
import { SectionHeading } from "./SectionHeading";

const metrics: { key: "accuracy" | "precision_macro" | "recall_macro" | "f1_macro"; label: string }[] = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision_macro", label: "Macro precision" },
  { key: "recall_macro", label: "Macro recall" },
  { key: "f1_macro", label: "Macro F1" },
];
const sentiments: Sentiment[] = ["negative", "neutral", "positive"];

function percentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function BarChart({
  title,
  description,
  series,
}: {
  title: string;
  description: string;
  series: { label: string; value: number; color: string }[];
}) {
  return (
    <article className="evaluation-chart">
      <h4>{title}</h4>
      <p>{description}</p>
      <div className="evaluation-chart-bars">
        {series.map((item) => (
          <div className="evaluation-chart-row" key={item.label}>
            <span>{item.label}</span>
            <div
              className="evaluation-chart-track"
              role="img"
              aria-label={`${item.label}: ${percentage(item.value)}`}
            >
              <span style={{ width: `${Math.max(0, Math.min(100, item.value * 100))}%`, backgroundColor: item.color }} />
            </div>
            <strong>{percentage(item.value)}</strong>
          </div>
        ))}
      </div>
    </article>
  );
}

function ConfusionMatrix({
  name,
  evaluation,
  labels,
}: {
  name: string;
  evaluation: SplitEvaluation;
  labels: Sentiment[];
}) {
  return (
    <article className="test-confusion-card">
      <h4>{name}</h4>
      <p>{evaluation.records.toLocaleString()} test records · actual × predicted</p>
      <div className="comparison-table-scroll">
        <table className="confusion-matrix" aria-label={`${name} final test confusion matrix`}>
          <thead>
            <tr><th scope="col">Actual</th>{labels.map((sentiment) => <th scope="col" key={sentiment}>{sentiment}</th>)}</tr>
          </thead>
          <tbody>
            {evaluation.confusion_matrix.map((row, rowIndex) => (
              <tr key={labels[rowIndex]}>
                <th scope="row">{labels[rowIndex]}</th>
                {row.map((count, columnIndex) => <td key={labels[columnIndex]}>{count.toLocaleString()}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

export function ModelLaboratorySection({
  statuses,
  evaluation,
  evaluationLoading,
  evaluationError,
  datasetAvailable,
  training,
  error,
  onTrain,
}: {
  statuses: ModelStatus[];
  evaluation: ModelEvaluation | null;
  evaluationLoading: boolean;
  evaluationError: string;
  datasetAvailable: boolean;
  training: boolean;
  error: string;
  onTrain: () => void;
}) {
  const modelResults = evaluation?.models ?? [];
  const labelOrder = evaluation?.label_order ?? [];
  const classAverages = sentiments.map((sentiment) => ({
    sentiment,
    average: modelResults.length
      ? modelResults.reduce((total, model) => total + model.test.per_class[sentiment].f1, 0) / modelResults.length
      : 0,
  }));
  const strongestClass = [...classAverages].sort((a, b) => b.average - a.average)[0];
  const weakestClass = [...classAverages].sort((a, b) => a.average - b.average)[0];
  const greatestF1Drop = [...modelResults].sort(
    (a, b) => (b.validation.f1_macro - b.test.f1_macro) - (a.validation.f1_macro - a.test.f1_macro),
  )[0];

  return (
    <section className="content-section laboratory-section" id="models">
      <SectionHeading
        title="Model Comparison"
        description="Validation results alongside a separately reported, held-out final test evaluation."
        action={
          <button className="button button-secondary train-button" type="button" onClick={onTrain} disabled={!datasetAvailable || training}>
            {training ? <><span className="button-spinner button-spinner-dark" /> Training…</> : "Train models"}
          </button>
        }
      />
      {error && <div className="request-error lab-request-error" role="alert">{error}</div>}
      {evaluationError && <div className="request-error lab-request-error" role="alert">Final evaluation unavailable: {evaluationError}</div>}
      {!datasetAvailable && (
        <div className="section-notice">Model evaluation is unavailable until TweetEval loads.</div>
      )}
      {evaluationLoading && (
        <div className="section-notice evaluation-loading" role="status">
          <span className="button-spinner button-spinner-dark" /> Evaluating the existing trained models on the held-out test split…
        </div>
      )}
      {!evaluation && !evaluationLoading && !evaluationError && datasetAvailable && (
        <div className="section-notice">Final held-out evaluation is not available yet. Ensure all three models are trained.</div>
      )}

      <div className="model-card-grid">
        {models.map((model) => {
          const status = statuses.find((item) => item.id === model.id);
          const evaluationModel = evaluation?.models.find((item) => item.id === model.id);
          const validation = evaluationModel?.validation;
          return (
            <article className={`research-model-card ${status?.trained ? "model-trained" : ""}`} key={model.id}>
              <div className="model-card-heading">
                <span className={`model-status ${status?.trained ? "model-status-trained" : status?.training ? "model-status-training" : ""}`}>
                  {status?.training ? "Training" : status?.trained ? "Trained" : status?.training_status === "failed" ? "Training failed" : "Not trained"}
                </span>
              </div>
              <h3>{model.name}</h3>
              <div className="model-specs">
                <div><span>Features</span><strong>{status?.feature_extraction ?? (model.id === "tfidf" ? "TF-IDF · unigrams" : model.id === "bigram" ? "Count · bigrams" : "Count · unigrams")}</strong></div>
                <div><span>Classifier</span><strong>{status?.algorithm ?? "Multinomial Naive Bayes"}</strong></div>
                <div><span>Smoothing</span><strong>{status?.smoothing ?? "Laplace · α = 1.0"}</strong></div>
              </div>
              <div className="model-metric-block">
                <div className="metric-block-heading"><span>Validation metrics</span><span>{validation ? `${validation.records.toLocaleString()} records` : "Not evaluated"}</span></div>
                <div className="model-metrics">
                  {metrics.map((metric) => (
                    <div className="model-metric" key={metric.key}>
                      <span>{metric.label}</span>
                      <strong>{validation ? percentage(validation[metric.key]) : "—"}</strong>
                    </div>
                  ))}
                </div>
              </div>
              {validation && (
                <div className="confusion-matrix-wrap">
                  <div className="confusion-matrix-title">Validation confusion matrix · actual × predicted</div>
                  <table className="confusion-matrix" aria-label={`${model.name} validation confusion matrix`}>
                    <thead>
                      <tr><th scope="col">Actual</th>{labelOrder.map((sentiment) => <th scope="col" key={sentiment}>{sentiment}</th>)}</tr>
                    </thead>
                    <tbody>
                      {validation.confusion_matrix.map((row, rowIndex) => (
                        <tr key={labelOrder[rowIndex]}>
                          <th scope="row">{labelOrder[rowIndex]}</th>
                          {row.map((count, columnIndex) => <td key={labelOrder[columnIndex]}>{count.toLocaleString()}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {status?.training_error && <div className="model-error">{status.training_error}</div>}
              <div className="model-card-footer"><span>{status?.trained_at ? `Trained ${new Date(status.trained_at).toLocaleDateString()}` : "Awaiting training"}</span></div>
            </article>
          );
        })}
      </div>

      {evaluation && (
        <>
          <div className="evaluation-summary">
            <div>
              <span>Top test Macro F1</span>
              <strong>{evaluation.best_model_by_macro_f1.model_names.join(" / ")}</strong>
              <small>{percentage(evaluation.best_model_by_macro_f1.score)}{evaluation.best_model_by_macro_f1.tie ? " · tied" : ""}</small>
            </div>
            <div>
              <span>Official test split</span>
              <strong>{evaluation.dataset_split_records.test.toLocaleString()} records</strong>
              <small>Separate from training and validation</small>
            </div>
            <div>
              <span>Test evaluation saved</span>
              <strong>{new Date(evaluation.evaluated_at).toLocaleString()}</strong>
              <small>Frozen model and dataset fingerprint</small>
            </div>
          </div>

          <div className="comparison-table-wrap evaluation-chart-wrap">
            <h3>Held-out performance charts</h3>
            <p className="evaluation-section-description">
              All values are computed from the same official test records; bars use the unrounded scores.
            </p>
            <div className="evaluation-chart-grid">
              {metrics.map((metric) => (
                <BarChart
                  key={metric.key}
                  title={`Test ${metric.label}`}
                  description="Higher is better · scale 0–100%"
                  series={evaluation.models.map((item, index) => ({
                    label: item.name,
                    value: item.test[metric.key],
                    color: ["var(--accent)", "var(--positive)", "var(--negative)"][index],
                  }))}
                />
              ))}
            </div>
            <div className="evaluation-chart-grid evaluation-chart-grid-bottom">
              <BarChart
                title="Macro F1 · validation versus test"
                description="Shows the observed split difference, not a significance test."
                series={evaluation.models.flatMap((item, index) => [
                  { label: `${item.name} · validation`, value: item.validation.f1_macro, color: "var(--muted)" },
                  { label: `${item.name} · test`, value: item.test.f1_macro, color: ["var(--accent)", "var(--positive)", "var(--warning)"][index] },
                ])}
              />
              <BarChart
                title="Test F1 by sentiment"
                description="Per-class F1 for every model · higher is better."
                series={evaluation.models.flatMap((item, index) =>
                  sentiments.map((sentiment) => ({
                    label: `${item.name} · ${sentiment}`,
                    value: item.test.per_class[sentiment].f1,
                    color: ["var(--accent)", "var(--positive)", "var(--warning)"][index],
                  })),
                )}
              />
            </div>
          </div>

          <div className="comparison-table-wrap">
            <h3>Validation and final test metrics</h3>
            <p className="evaluation-section-description">
              The signed delta is test minus validation. Test metrics never replace the validation metrics stored with model training.
            </p>
            <div className="comparison-table-scroll">
              <table className="comparison-table">
                <thead>
                  <tr><th scope="col">Model</th><th scope="col">Split</th><th scope="col">Records</th><th scope="col">Accuracy</th><th scope="col">Macro precision</th><th scope="col">Macro recall</th><th scope="col">Macro F1</th><th scope="col">Δ Macro F1</th></tr>
                </thead>
                <tbody>
                  {evaluation.models.map((item) => (
                    <Fragment key={item.id}>
                      <tr>
                        <th scope="row" rowSpan={2}>{item.name}</th>
                        <td>Validation</td>
                        <td>{item.validation.records.toLocaleString()}</td>
                        <td>{percentage(item.validation.accuracy)}</td>
                        <td>{percentage(item.validation.precision_macro)}</td>
                        <td>{percentage(item.validation.recall_macro)}</td>
                        <td>{percentage(item.validation.f1_macro)}</td>
                        <td rowSpan={2} className={item.test_minus_validation.f1_macro < 0 ? "evaluation-delta-negative" : "evaluation-delta-positive"}>
                          {item.test_minus_validation.f1_macro > 0 ? "+" : ""}{(item.test_minus_validation.f1_macro * 100).toFixed(2)} pp
                        </td>
                      </tr>
                      <tr>
                        <td>Final test</td>
                        <td>{item.test.records.toLocaleString()}</td>
                        <td>{percentage(item.test.accuracy)}</td>
                        <td>{percentage(item.test.precision_macro)}</td>
                        <td>{percentage(item.test.recall_macro)}</td>
                        <td>{percentage(item.test.f1_macro)}</td>
                      </tr>
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="comparison-table-wrap">
            <h3>Per-class final test report</h3>
            <p className="evaluation-section-description">Precision, recall, F1 and true support for each sentiment class.</p>
            <div className="comparison-table-scroll">
              <table className="comparison-table">
                <thead><tr><th scope="col">Sentiment</th><th scope="col">Model</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1-score</th><th scope="col">Support</th></tr></thead>
                <tbody>
                  {sentiments.flatMap((sentiment) => evaluation.models.map((item) => {
                    const values = item.test.per_class[sentiment];
                    return (
                      <tr key={`${sentiment}-${item.id}`}>
                        <th scope="row">{sentiment}</th>
                        <td>{item.name}</td>
                        <td>{percentage(values.precision)}</td>
                        <td>{percentage(values.recall)}</td>
                        <td>{percentage(values.f1)}</td>
                        <td>{values.support.toLocaleString()}</td>
                      </tr>
                    );
                  }))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="test-confusion-grid">
            {evaluation.models.map((item) => (
              <ConfusionMatrix
                key={item.id}
                name={item.name}
                evaluation={item.test}
                labels={evaluation.label_order}
              />
            ))}
          </div>

          <div className="evaluation-insight" role="note">
            <strong>Observed from these test results:</strong>{" "}
            {strongestClass && weakestClass && (
              <>the highest mean class F1 across the three models is {strongestClass.sentiment} ({percentage(strongestClass.average)}), while the lowest is {weakestClass.sentiment} ({percentage(weakestClass.average)}). </>
            )}
            {greatestF1Drop && (
              <>The largest validation-to-test Macro F1 decrease is {greatestF1Drop.name} ({(greatestF1Drop.test_minus_validation.f1_macro * 100).toFixed(2)} percentage points). These aggregate results do not establish statistical significance.</>
            )}
          </div>
          <div className="evaluation-note">
            Test predictions were generated once for each frozen model and saved as a separate final-evaluation artifact. They were not used for fitting or tuning. Per-class support sums to {evaluation.models[0]?.test.records.toLocaleString()} records.
          </div>
        </>
      )}

      <div className="evaluation-note">
        Validation metrics are computed during training. Final test results, when available, are stored separately and do not overwrite validation results.
      </div>
    </section>
  );
}
