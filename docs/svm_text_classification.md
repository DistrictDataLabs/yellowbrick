# SVM Text Classification Example

## Overview

This contribution adds a self-contained text-classification example to the
Yellowbrick project. The example uses Term Frequency-Inverse Document
Frequency (TF-IDF) features and a linear Support Vector Machine (SVM) to
classify short text messages into three sentiment categories:

- Positive
- Negative
- Neutral

Yellowbrick classification visualizers are used to evaluate and interpret the
model's performance.

## Objectives

The objectives of this example are to:

1. Demonstrate how text can be converted into numerical features using
   `TfidfVectorizer`.
2. Train a multiclass linear SVM classifier.
3. Evaluate classification performance using accuracy and a classification
   report.
4. Visualize model performance using Yellowbrick's `ConfusionMatrix` and
   `ClassificationReport` visualizers.
5. Provide a reproducible example with tests and documentation.

## Requirements

The example requires Python and the following packages:

- scikit-learn
- Yellowbrick
- pandas
- matplotlib
- pytest for running tests

The example was tested using Python 3.11 because the repository's current
dependencies include older compatibility requirements.

## Running the Example

From the root directory of the Yellowbrick repository, activate the virtual
environment and run:

```bash
python examples/svm_text_classification.py
```

On Windows PowerShell, the equivalent command is:

```powershell
python examples\svm_text_classification.py
```

The program prints the following information:

- Dataset shape
- Sentiment-class distribution
- Model accuracy
- Precision, recall, and F1-score
- Predictions for new text messages

## Methodology

### Dataset

The example uses a small, self-contained dataset of manually written text
messages. Each message has one of three labels: positive, negative, or
neutral. The dataset is included directly in the example so that users do not
need to download an external file.

### Feature Extraction

The text is converted into numerical features using
`TfidfVectorizer`. Both individual words and two-word combinations are used
through unigram and bigram features. Lowercasing is applied during
preprocessing.

### Classification Model

A `LinearSVC` model is trained on the TF-IDF feature matrix. A linear SVM is a
reasonable baseline for text classification because text data often produces
high-dimensional, sparse feature matrices.

### Evaluation

The dataset is divided into training and testing subsets using a stratified
split. Stratification ensures that each sentiment class is represented in
both subsets. The model is evaluated using accuracy, precision, recall,
F1-score, and support.

## Yellowbrick Visualizations

The example creates two visualizations:

### Confusion Matrix

The confusion matrix shows the relationship between the true sentiment labels
and the labels predicted by the SVM. Correct predictions appear on the
diagonal, while incorrect predictions appear outside the diagonal.

### Classification Report

The classification-report visualizer displays precision, recall, and F1-score
for each sentiment category. These measures provide more information than
accuracy alone, particularly when model performance differs between classes.

The generated plots are saved in the `outputs` directory:

```text
outputs/confusion_matrix.png
outputs/classification_report.png
```

## Testing

Unit tests are provided in:

```text
tests/test_svm_text_classification.py
```

Run the tests with:

```bash
pytest -q tests/test_svm_text_classification.py
```

The tests verify that:

- The model can generate predictions.
- The number of predictions matches the number of input messages.
- The model contains the expected TF-IDF and SVM components.

The test suite completed successfully with three passing tests.

## Reproducibility

The example uses a fixed random seed of `42`. This makes the train-test split
reproducible when the program is run multiple times.

The dataset is self-contained, and the preprocessing and classification steps
are implemented directly in the example file. The generated plots and
evaluation results can therefore be recreated by running the same command in
the same software environment.

## Technical Difficulties and Resolutions

### Python Compatibility

The initial setup failed because the repository imported
`distutils.version`, but the `distutils` module was removed from Python 3.12
and later. The issue was resolved by creating a Python 3.11 virtual
environment and reinstalling the project dependencies.

### Yellowbrick Estimator Detection

The initial implementation passed a scikit-learn pipeline directly to the
Yellowbrick visualizers. Yellowbrick did not recognize the pipeline as a
classifier. The text vectorizer and SVM were therefore separated so that the
visualizers received the trained `LinearSVC` model directly.

The installed versions of Yellowbrick and scikit-learn also produced an
estimator-type compatibility issue. The `force_model=True` option was used in
the Yellowbrick visualizers to bypass the outdated type check while retaining
the normal classifier interface.

### Testing Dependencies

The repository's `setup.cfg` referenced pytest coverage and specification
plugins that were not initially installed. The missing testing dependencies
were installed before running the tests.

## Best Practices Identified

The following best practices were identified during the implementation:

1. Use a virtual environment to isolate project dependencies.
2. Check the repository's supported Python version before installing it.
3. Use a fixed random seed when demonstrating machine-learning results.
4. Use stratified splitting for multiclass classification tasks.
5. Keep feature extraction and model training clearly separated.
6. Include unit tests for model behavior and expected output structure.
7. Document installation, execution, limitations, and known compatibility
   issues.
8. Do not rely on accuracy alone; use class-level precision, recall, F1-score,
   and visual diagnostics.

## Limitations

The dataset is intentionally small and manually created for demonstration
purposes. The resulting accuracy should not be interpreted as evidence of
real-world sentiment-analysis performance.

The dataset does not represent the full variety of language used in real
messages. The model may perform poorly on text containing vocabulary,
grammar, sarcasm, spelling variations, or sentence patterns that are not
present in the training data.

A production system would require a substantially larger, independently
labeled dataset, more extensive validation, bias analysis, and monitoring of
model performance after deployment.

## Contribution Summary

This contribution provides a reproducible SVM text-classification example
that demonstrates how Yellowbrick can be used to visualize the performance of
a multiclass machine-learning classifier. It includes executable code,
automated tests, generated evaluation plots, usage instructions, technical
difficulty documentation, and limitations.
