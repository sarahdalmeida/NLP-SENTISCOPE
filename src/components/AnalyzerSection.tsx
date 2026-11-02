import { exampleTexts, models } from "../data";
import type { ModelId } from "../types";
import { SectionHeading } from "./SectionHeading";

export function AnalyzerSection({
  text,
  model,
  loading,
  error,
  onTextChange,
  onModelChange,
  onAnalyze,
  onClear,
}: {
  text: string;
  model: ModelId;
  loading: boolean;
  error: string;
  onTextChange: (value: string) => void;
  onModelChange: (value: ModelId) => void;
  onAnalyze: () => void;
  onClear: () => void;
}) {
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  return (
    <section className="content-section analyzer-section" id="analyser">
      <SectionHeading title="Sentiment Analysis" description="Classify text with a trained Naive Bayes model." />
      <div className="analyser-card">
        <div className="analyser-main">
          <label className="textarea-label" htmlFor="sentiment-text">Text to analyse</label>
          <textarea
            id="sentiment-text"
            className="sentiment-input"
            maxLength={5000}
            placeholder="Enter text here…"
            value={text}
            onChange={(event) => onTextChange(event.target.value)}
          />
          <div className="input-footer">
            <span>{text.length} / 5,000 characters · {wordCount} {wordCount === 1 ? "word" : "words"}</span>
            <button type="button" className="clear-action" disabled={!text && !error} onClick={onClear}>Clear</button>
          </div>
          <div className="example-picker">
            <div className="example-list">
              {exampleTexts.map((example, index) => (
                <button
                  type="button"
                  className={`example-chip example-${["positive", "negative", "neutral"][index]}`}
                  key={example.label}
                  onClick={() => onTextChange(example.text)}
                >
                  {["Positive", "Negative", "Neutral"][index]}
                </button>
              ))}
            </div>
          </div>
        </div>
        <aside className="analyser-controls">
          <label htmlFor="model-choice" className="select-label">Model</label>
          <select
            id="model-choice"
            className="model-select"
            value={model}
            onChange={(event) => onModelChange(event.target.value as ModelId)}
          >
            {models.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}
          </select>
          <p className="selected-model-note">{models.find((item) => item.id === model)?.features}</p>
          <button type="button" className="button button-primary analyse-button" onClick={onAnalyze} disabled={!text.trim() || loading}>
            {loading ? <><span className="button-spinner" /> Analysing…</> : "Analyse"}
          </button>
          <p className="analyser-note">Analysis uses the configured backend.</p>
          {error && <div className="request-error" role="alert">{error}</div>}
        </aside>
      </div>
    </section>
  );
}
