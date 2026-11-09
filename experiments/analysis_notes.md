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

### Future Investigation

- Compare corrected, regressed, and commonly misclassified samples.
- Evaluate whether the observed patterns persist across different random seeds.
- Investigate whether CNNs improve recognition of ambiguous digit shapes.

---

## 2. Hyperparameter Experiments

*To be completed.*

## 3. CNN Comparison

*To be completed.*