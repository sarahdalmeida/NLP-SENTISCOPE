import { useCallback, useEffect, useRef, useState } from "react";
import {
  analyzeText,
  getDatasetPreview,
  getDatasetSummary,
  getHealth,
  getModelEvaluation,
  reloadDataset,
  trainModels,
} from "./services/api";
import type {
  AnalysisResult,
  BackendHealth,
  DatasetPreview,
  DatasetSummary,
  ModelEvaluation,
  ModelId,
} from "./types";
import { AnalyzerSection } from "./components/AnalyzerSection";
import { DashboardSection } from "./components/DashboardSection";
import { DatasetExplorerSection } from "./components/DatasetExplorerSection";
import { Hero } from "./components/Hero";
import { MethodologySection } from "./components/MethodologySection";
import { ModelLaboratorySection } from "./components/ModelLaboratorySection";
import { ResultsSection } from "./components/ResultsSection";

function App() {
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [healthError, setHealthError] = useState("");
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [datasetError, setDatasetError] = useState("");
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [label, setLabel] = useState("all");
  const [text, setText] = useState("");
  const [model, setModel] = useState<ModelId>("unigram");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [analysisError, setAnalysisError] = useState("");
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [trainingError, setTrainingError] = useState("");
  const [evaluation, setEvaluation] = useState<ModelEvaluation | null>(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");
  const [activeSection, setActiveSection] = useState(() => window.location.hash.slice(1) || "dashboard");
  const analysisRequestId = useRef(0);
  const datasetAvailable = Boolean(health?.dataset.available);
  const trainedModelSignature = (health?.models ?? [])
    .map((item) => `${item.id}:${item.trained_at ?? item.training_status}`)
    .join("|");

  const refreshHealth = useCallback(async () => {
    try {
      const currentHealth = await getHealth();
      setHealth(currentHealth);
      setHealthError("");
      return currentHealth;
    } catch (error) {
      setHealthError(error instanceof Error ? error.message : "The backend is unavailable.");
      return null;
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  useEffect(() => {
    if (health?.dataset.status !== "loading") return;
    const timer = window.setInterval(() => {
      void refreshHealth();
    }, 2000);
    return () => window.clearInterval(timer);
  }, [health?.dataset.status, refreshHealth]);

  useEffect(() => {
    if (!datasetAvailable) {
      setSummary(null);
      setPreview(null);
      setDatasetLoading(false);
      return;
    }
    let active = true;
    setDatasetError("");
    getDatasetSummary()
      .then((nextSummary) => {
        if (!active) return;
        setSummary(nextSummary);
      })
      .catch((error: unknown) => {
        if (active) {
          setDatasetError(error instanceof Error ? error.message : "Unable to load dataset details.");
        }
      })
    return () => {
      active = false;
    };
  }, [datasetAvailable, health?.dataset.total_records]);

  useEffect(() => {
    if (!datasetAvailable) return;
    let active = true;
    const timer = window.setTimeout(() => {
      setDatasetLoading(true);
      void getDatasetPreview({ q: query, label })
        .then((nextPreview) => {
          if (active) setPreview(nextPreview);
        })
        .catch((error: unknown) => {
          if (active) {
            setDatasetError(error instanceof Error ? error.message : "Unable to search dataset records.");
          }
        })
        .finally(() => {
          if (active) setDatasetLoading(false);
        });
    }, 220);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [datasetAvailable, label, query]);

  useEffect(() => {
    const updateActiveSection = () => {
      const section = window.location.hash.slice(1);
      setActiveSection(["dashboard", "analyser", "models", "dataset", "methodology"].includes(section) ? section : "dashboard");
    };
    window.addEventListener("hashchange", updateActiveSection);
    return () => window.removeEventListener("hashchange", updateActiveSection);
  }, []);

  useEffect(() => {
    document.getElementById(activeSection)?.scrollIntoView({
      block: "start",
      behavior: window.location.hash === `#${activeSection}` ? "instant" : "smooth",
    });
  }, [activeSection]);

  useEffect(() => {
    if (activeSection !== "models") return;
    if (
      !datasetAvailable ||
      !health?.models.length ||
      health.models.some((item) => !item.trained)
    ) {
      setEvaluation(null);
      setEvaluationError("");
      setEvaluationLoading(false);
      return;
    }
    let active = true;
    setEvaluation(null);
    setEvaluationError("");
    setEvaluationLoading(true);
    getModelEvaluation()
      .then((result) => {
        if (active) setEvaluation(result);
      })
      .catch((error: unknown) => {
        if (active) {
          setEvaluationError(error instanceof Error ? error.message : "Final model evaluation is unavailable.");
        }
      })
      .finally(() => {
        if (active) setEvaluationLoading(false);
      });
    return () => {
      active = false;
    };
  }, [activeSection, datasetAvailable, health?.models, trainedModelSignature]);

  const handleAnalyze = async () => {
    const requestId = ++analysisRequestId.current;
    const input = text.trim();
    if (!input) {
      setResult(null);
      setAnalysisError("Enter text before starting sentiment analysis.");
      setAnalysisLoading(false);
      return;
    }
    if (input.length > 5000) {
      setResult(null);
      setAnalysisError("Text must be 5,000 characters or fewer.");
      setAnalysisLoading(false);
      return;
    }
    setAnalysisLoading(true);
    setResult(null);
    setAnalysisError("");
    try {
      const response = await analyzeText(input, model);
      if (requestId !== analysisRequestId.current) return;
      setResult(response);
      window.setTimeout(() => {
        document.getElementById("results")?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 50);
    } catch (error) {
      if (requestId === analysisRequestId.current) {
        setAnalysisError(error instanceof Error ? error.message : "The analysis request failed.");
      }
    } finally {
      if (requestId === analysisRequestId.current) setAnalysisLoading(false);
    }
  };

  const handleTrain = async () => {
    setTraining(true);
    setTrainingError("");
    try {
      const response = await trainModels();
      setHealth((current) => current ? { ...current, models: response.models } : current);
      const failures = response.models.filter((item) => item.training_status === "failed");
      if (failures.length) {
        setTrainingError(failures.map((item) => `${item.name}: ${item.training_error}`).join(" "));
      } else {
        setTrainingError("");
      }
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "Model training failed.");
      void refreshHealth();
    } finally {
      setTraining(false);
    }
  };

  const handleDatasetRetry = async () => {
    setDatasetError("");
    setHealth((current) => current
      ? { ...current, dataset: { ...current.dataset, status: "loading", error: null } }
      : current);
    try {
      await reloadDataset();
      await refreshHealth();
    } catch (error) {
      setDatasetError(error instanceof Error ? error.message : "Dataset reload failed.");
      await refreshHealth();
    }
  };

  return (
    <>
      <a className="skip-link" href="#analyser">Skip to sentiment analysis</a>
      <Hero activeSection={activeSection} onNavigate={setActiveSection} />
      {healthError && <div className="global-api-alert" role="status"><span>{healthError}</span><button onClick={() => void refreshHealth()}>Retry connection</button></div>}
      <main>
        <DashboardSection health={health} summary={summary} healthError={healthError} />
        <AnalyzerSection
          text={text}
          model={model}
          loading={analysisLoading}
          error={analysisError}
          onTextChange={(value) => {
            analysisRequestId.current += 1;
            setText(value);
            setResult(null);
            setAnalysisError("");
            setAnalysisLoading(false);
          }}
          onModelChange={(value) => {
            analysisRequestId.current += 1;
            setModel(value);
            setResult(null);
            setAnalysisError("");
            setAnalysisLoading(false);
          }}
          onAnalyze={() => void handleAnalyze()}
          onClear={() => {
            analysisRequestId.current += 1;
            setText("");
            setResult(null);
            setAnalysisError("");
            setAnalysisLoading(false);
          }}
        />
        <ResultsSection result={result} loading={analysisLoading} error={analysisError} />
        <ModelLaboratorySection
          statuses={health?.models ?? []}
          evaluation={evaluation}
          evaluationLoading={evaluationLoading}
          evaluationError={evaluationError}
          datasetAvailable={datasetAvailable}
          training={training}
          error={trainingError}
          onTrain={() => void handleTrain()}
        />
        <DatasetExplorerSection
          health={health}
          summary={summary}
          preview={preview}
          query={query}
          label={label}
          loading={datasetLoading}
          error={datasetError}
          onQueryChange={setQuery}
          onLabelChange={setLabel}
          onRetry={() => void handleDatasetRetry()}
        />
        <MethodologySection />
      </main>
      <footer className="site-footer"><span>SentiScope · Academic NLP project</span><a href="https://huggingface.co/datasets/cardiffnlp/tweet_eval" target="_blank" rel="noreferrer">TweetEval dataset</a></footer>
    </>
  );
}

export default App;
