export type ModelName = "dense" | "rnn" | "lstm" | "gru" | "bert";
export type PredictionLabel = "REAL" | "FAKE" | "UNCERTAIN";

export interface ModelInfo {
  name: ModelName;
  available: boolean;
  parameter_count: number | null;
  description: string;
}

export interface Prediction {
  prediction: PredictionLabel;
  confidence: number;
  real_probability: number;
  fake_probability: number;
  model: string;
  explanation_note: string;
}

export interface Explanation {
  model: string;
  prediction: string;
  words: Array<{ word: string; weight: number }>;
  note: string;
}

export interface Metrics {
  model: string;
  parameter_count: number;
  test_accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc?: number;
  confusion_matrix?: number[][];
  history?: Record<string, number[]>;
  hyperparameters?: Record<string, unknown>;
}

export type ExperimentCurves = Record<string, Record<string, number[]>>;

export interface DatasetStats {
  total: number;
  real_count: number;
  fake_count: number;
  class_distribution: Record<string, number>;
  article_length_words: Record<string, number>;
  headline_length_words: Record<string, number>;
  split_counts: Record<string, number>;
}

export interface ApiError {
  error: string;
  code: string;
}
