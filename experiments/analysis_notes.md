# Experiment Analysis Notes

This document records observations, interpretations, and hypotheses from the MNIST architecture experiments.

## 1. Softmax Regression vs. MLP

### Key Findings

- **Validation accuracy:** Softmax Regression achieved 90.92%, while the MLP achieved 92.50%, an improvement of 1.58 percentage points.
- **Per-class performance:** The largest improvement was observed for digit 8, where recall increased from 86.48% to 91.65%.
- **Error patterns:** The number of digit 8 samples misclassified as 5 decreased from 16 to 3.
- **Uneven improvement:** Performance gains were not equally distributed across all classes. Digit 1 showed no improvement, while digit 4 showed a minor decrease of 0.20 percentage points.

### Hypothesis — Improved Recognition of Ambiguous Digit Shapes

The MLP demonstrates a substantial improvement in recall for digit 8 compared with Softmax Regression (91.65% vs. 86.48%). Notably, the number of digit 8 samples misclassified as 5 decreases from 16 to 3.

Visual inspection of selected misclassified samples reveals unusual stroke patterns, partially open loops, and shapes resembling other digits.

One possible explanation is that the MLP's nonlinear hidden layer enables it to learn more complex combinations of pixel patterns, whereas Softmax Regression relies on linear class scores.

However, this remains a hypothesis rather than a demonstrated explanation of the learned representations. The visual observations are based on a limited sample, and further controlled experiments would be required to identify which features contribute to the improvement.

### Model Complexity and Training Time

The MLP contains 101,770 trainable parameters, approximately 12.96 times more than Softmax Regression (7,850 parameters).

In the initial benchmark, Softmax Regression completed training in 4.53 seconds, compared with 46.66 seconds for the MLP (approximately 10.30 times longer).

Despite the additional computational cost, the MLP improved validation accuracy by 1.58 percentage points and reduced validation cross-entropy loss by approximately 19.9%.

These findings illustrate a trade-off between predictive performance and computational cost. Training times were measured in a single run and should be interpreted as preliminary benchmarks rather than definitive performance measurements.

## 2. Hyperparameter Experiments

*To be completed.*

## 3. CNN Comparison

*To be completed.*