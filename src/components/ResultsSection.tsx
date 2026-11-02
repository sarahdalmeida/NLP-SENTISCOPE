import type { AnalysisResult, Sentiment } from "../types";
import { SectionHeading } from "./SectionHeading";

const sentimentLabels: { id: Sentiment; name: string; tone: string }[] = [
  { id: "positive", name: "Positive", tone: "positive" },
  { id: "neutral", name: "Neutral", tone: "neutral" },
  { id: "negative", name: "Negative", tone: "negative" },
];

export function ResultsSection({
  result,
  loading,
  error,
}: {
  result: AnalysisResult | null;
  loading: boolean;
  error: string;
}) {
  const supportingFeatures = result?.explanations.filter((item) => item.effect === "supports") ?? [];
  const opposingFeatures = result?.explanations.filter((item) => item.effect === "opposes") ?? [];
  const featureType = result?.model_id === "bigram" ? "two-word phrases" : "words";

  return (
    <section className="content-section results-section" id="results" aria-live="polite">
      <SectionHeading title="Prediction Results" description="Class prediction and probabilities returned by the selected model." />
      {!result ? (
        <div className={`results-empty ${loading ? "results-loading" : ""}`}>
          <p>{loading ? "Analysing…" : error ? "No result. Resolve the error above and try again." : "Results will appear here after analysis."}</p>
        </div>
      ) : (
        <div className="result-grid">
          <div className={`prediction-card prediction-${result.predicted_sentiment}`}>
            <h3>Predicted sentiment</h3>
            <div className="prediction-label">{result.predicted_sentiment}</div>
            <p>Confidence {(result.confidence * 100).toFixed(1)}%</p>
            <p>Model: {result.model_name}</p>
          </div>
          <div className="probability-card">
            <h3>Class probabilities</h3>
            <div className="probability-list">
              {sentimentLabels.map(({ id, name, tone }) => (
                <div className={`probability-row probability-${tone}`} key={id}>
                  <div className="probability-label">{name}</div>
                  <div className="probability-track"><span style={{ width: `${Math.min(100, result.probabilities[id] * 100)}%` }} /></div>
                  <strong>{(result.probabilities[id] * 100).toFixed(1)}%</strong>
                </div>
              ))}
            </div>
          </div>
          <div className="contribution-card">
            <h3>Feature contributions</h3>
            <div className="contribution-results">
              {result.metadata.explanation_available ? (
                result.explanations.length ? (
                  <>
                    <p className="contribution-note">
                      Showing {featureType} from {result.model_name}. Scores compare feature evidence with the strongest alternative class.
                    </p>
                    {[
                      { title: "Supports predicted sentiment", features: supportingFeatures, effect: "supports" },
                      { title: "Opposes predicted sentiment", features: opposingFeatures, effect: "opposes" },
                    ].map(({ title, features, effect }) => (
                      <div className="contribution-group" key={effect}>
                        <h4>{title}</h4>
                        {features.length ? (
                          <ul className="contribution-list">
                            {features.map((item) => (
                              <li className={`contribution-${item.effect}`} key={`${item.feature}-${item.effect}`}>
                                <span>{item.feature}</span>
                                <strong>{item.score.toFixed(3)}</strong>
                              </li>
                            ))}
                          </ul>
                        ) : <p className="contribution-empty">No non-zero feature contributions in this direction.</p>}
                      </div>
                    ))}
                  </>
                ) : <p>No non-zero feature contributions were available for this input and class comparison.</p>
              ) : <p>Contribution scores are not available from the current backend. No feature-level explanation is inferred.</p>}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
