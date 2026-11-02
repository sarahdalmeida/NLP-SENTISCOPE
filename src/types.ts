export type ModelId = "unigram" | "bigram" | "tfidf";

export type Sentiment = "positive" | "negative" | "neutral";

export interface ModelDefinition {
  id: ModelId;
  name: string;
  shortName: string;
  description: string;
  features: string;
  weighting: string;
}

export interface AnalysisResult {
  predicted_sentiment: Sentiment;
  model_id: ModelId;
  model_name: string;
  model_status: "trained";
  confidence: number;
  probabilities: Record<Sentiment, number>;
  explanations: {
    feature: string;
    score: number;
    effect: "supports" | "opposes";
  }[];
  metadata: {
    source: string;
    explanation_available: boolean;
  };
}

export interface ModelStatus {
  id: ModelId;
  name: string;
  feature_extraction: string;
  algorithm: string;
  smoothing: string;
  trained: boolean;
  training: boolean;
  training_status: "training" | "trained" | "failed" | "not_trained";
  training_error: string | null;
  evaluation_status: "validation" | "not_evaluated";
  evaluation: {
    split: string;
    records: number;
    accuracy: number;
    precision_macro: number;
    recall_macro: number;
    f1_macro: number;
    confusion_matrix: number[][];
    labels: Sentiment[];
  } | null;
  trained_at: string | null;
}

export interface SplitEvaluation {
  split: "validation" | "test";
  records: number;
  correct: number;
  incorrect: number;
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  per_class: Record<Sentiment, {
    precision: number;
    recall: number;
    f1: number;
    support: number;
  }>;
  confusion_matrix: number[][];
  classification_report: Record<string, number | Record<string, number>>;
}

export interface ModelEvaluation {
  evaluation_id: string;
  evaluation_version: string;
  status: "complete";
  dataset: string;
  configuration: string;
  dataset_split_records: Record<"train" | "validation" | "test", number>;
  label_order: Sentiment[];
  test_split_usage: "final_evaluation_only";
  evaluated_at: string;
  models: {
    id: ModelId;
    name: string;
    feature_extraction: string;
    training_split: "train";
    training_records: number;
    trained_at: string;
    validation: SplitEvaluation;
    test: SplitEvaluation;
    test_minus_validation: Record<"accuracy" | "precision_macro" | "recall_macro" | "f1_macro", number>;
  }[];
  best_model_by_macro_f1: {
    score: number;
    model_ids: ModelId[];
    model_names: string[];
    tie: boolean;
  };
}

export interface DatasetSummary {
  name: string;
  source: string;
  configuration: string;
  source_url: string;
  total_records: number;
  splits: Record<"train" | "validation" | "test", number>;
  label_mapping: Record<string, Sentiment>;
  distribution: Record<Sentiment, { count: number; proportion: number }>;
  quality: {
    empty_text_records: number;
    duplicate_text_records: number;
    label_values_valid: boolean;
  };
}

export interface DatasetPreview {
  dataset: string;
  split: string;
  offset: number;
  limit: number;
  total_matches: number;
  records: {
    id: string;
    split: string;
    text: string;
    label: number;
    sentiment: Sentiment;
  }[];
}

export interface BackendHealth {
  status: string;
  dataset: {
    status: "not_loaded" | "loading" | "available" | "unavailable";
    available: boolean;
    total_records: number | null;
    error: string | null;
  };
  models: ModelStatus[];
}
