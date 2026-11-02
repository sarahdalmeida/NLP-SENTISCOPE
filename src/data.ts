import type { ModelDefinition } from "./types";

export const models: ModelDefinition[] = [
  {
    id: "unigram",
    name: "Unigram Naive Bayes",
    shortName: "Unigram NB",
    description: "A clear baseline that learns from individual word frequencies.",
    features: "Single-token counts",
    weighting: "Term frequency",
  },
  {
    id: "bigram",
    name: "Bigram Naive Bayes",
    shortName: "Bigram NB",
    description: "Adds neighboring word pairs to capture short phrases and negation.",
    features: "Two-token sequences",
    weighting: "Term frequency",
  },
  {
    id: "tfidf",
    name: "TF-IDF + Naive Bayes",
    shortName: "TF-IDF + NB",
    description: "Weights informative terms before fitting a probabilistic classifier.",
    features: "Single-token TF-IDF",
    weighting: "TF-IDF",
  },
];

export const exampleTexts: { label: string; text: string }[] = [
  {
    label: "A welcoming experience",
    text: "The staff made us feel welcome, and every detail was thoughtful. I would happily return.",
  },
  {
    label: "A disappointing purchase",
    text: "It arrived late and stopped working after one day. Customer support never replied.",
  },
  {
    label: "A straightforward observation",
    text: "The parcel arrived on Tuesday. It contained the items listed in my order.",
  },
];
