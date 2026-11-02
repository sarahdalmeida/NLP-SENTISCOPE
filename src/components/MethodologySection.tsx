import { SectionHeading } from "./SectionHeading";

const steps = [
  ["Text preprocessing", "Normalize case and whitespace; replace URLs and mentions; preserve sentiment-bearing words and negation."],
  ["Tokenisation", "Convert normalized text into tokens using the same fitted vectorizer at training and prediction time."],
  ["Unigram feature extraction", "Count individual word features for the unigram Naive Bayes model."],
  ["Bigram feature extraction", "Count adjacent word pairs to retain short phrase context, including negation."],
  ["TF-IDF feature extraction", "Weight unigram counts by inverse document frequency learned from training data."],
  ["Multinomial Naive Bayes", "Estimate class probabilities from feature counts under the conditional-independence assumption."],
  ["Laplace smoothing", "Use α = 1 additive smoothing so unseen features do not yield zero likelihood."],
  ["Prediction and evaluation", "Return one of three sentiment classes and probabilities; report metrics on validation data."],
];

export function MethodologySection() {
  return (
    <section className="content-section methodology-section" id="methodology">
      <SectionHeading title="Methodology" description="The pipeline used by the current scikit-learn models." />
      <ol className="methodology-list">
        {steps.map(([title, text]) => <li className="methodology-step" key={title}><h3>{title}</h3><p>{text}</p></li>)}
      </ol>
    </section>
  );
}
