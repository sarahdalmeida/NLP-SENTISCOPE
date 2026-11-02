from typing import Any, Literal

from pydantic import BaseModel, Field

ModelId = Literal["unigram", "bigram", "tfidf"]
SentimentName = Literal["negative", "neutral", "positive"]
EvaluationSplitName = Literal["validation", "test"]


class AnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    model: ModelId


class AnalysisMetadata(BaseModel):
    source: str = "trained_multinomial_naive_bayes"
    explanation_available: bool = False


class FeatureContribution(BaseModel):
    feature: str
    score: float
    effect: Literal["supports", "opposes"]


class AnalysisResponse(BaseModel):
    predicted_sentiment: SentimentName
    model_id: ModelId
    model_name: str
    model_status: Literal["trained"]
    probabilities: dict[SentimentName, float]
    confidence: float = Field(ge=0, le=1)
    explanations: list[FeatureContribution]
    metadata: AnalysisMetadata


class TrainRequest(BaseModel):
    force: bool = False


class PerClassEvaluation(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class SplitEvaluation(BaseModel):
    split: EvaluationSplitName
    records: int
    correct: int
    incorrect: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class: dict[SentimentName, PerClassEvaluation]
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]


class EvaluatedModel(BaseModel):
    id: ModelId
    name: str
    feature_extraction: str
    training_split: Literal["train"]
    training_records: int
    trained_at: str
    validation: SplitEvaluation
    test: SplitEvaluation
    test_minus_validation: dict[str, float]


class BestModelByF1(BaseModel):
    score: float
    model_ids: list[ModelId]
    model_names: list[str]
    tie: bool


class ModelEvaluationResponse(BaseModel):
    evaluation_id: str
    evaluation_version: str
    status: Literal["complete"]
    dataset: str
    configuration: str
    dataset_split_records: dict[Literal["train", "validation", "test"], int]
    label_order: list[SentimentName]
    test_split_usage: Literal["final_evaluation_only"]
    evaluated_at: str
    models: list[EvaluatedModel]
    best_model_by_macro_f1: BestModelByF1
