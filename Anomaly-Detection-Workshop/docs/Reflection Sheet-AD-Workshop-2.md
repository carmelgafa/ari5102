
# Anomaly Detection Workshop 2 – Reflection Sheet

**Name(s):** Carmel Gafa  **Date:** 20.05.2025

Instructions
*Work individually or in pairs. Answer concisely (bullet points are fine). Save this file (or export to PDF) and email it to the instructor by **18:00 tomorrow**. Feel free to reference code, figures, or metrics from your notebook runs.*

---

## Notebook 01 – Dynamic (Time Series) Anomaly Detection

1. **Anomaly Type Identification**
**Describe one example from the notebook where a contextual anomaly was detected that would have been missed using a simple threshold method.**

Let us start by rephrasing the characteristics of contextual anomalies, that is a data point appears normal when considered alone (i.e. its value isn't extreme),but is anomalous when considered in the context of its temporal sequence or neighboring points.

The `generate_synthetic_data` function contained a specific section that generated a contextual anomaly, however a couple of issues were noted:

- The anomaly was meant to start between 0:00 and 6:00 (unusually high value at night), but it was specified to start at index 400, that is 16:00. The index was moved to 600, that is 0:00 so that it would start at an appropriate time of night.
- The anomaly offset the value by a value of 15, which was excessively large, even for the daytime and would have exceeded the threshold. It was was adjusted to a value of 6 so that the anomaly would be less extreme.

In order to facilitate the process, point and collective anomalies were temporary commented out to visualize the contextual anomaly only, obtaining the following figure:

![Contextual Anomaly](img-2-1-1.png)

The `calculate_metrics` function was modified slightly to include a confusion matrix. The function yielded the following results:

```text
confusion matrix:
[[995   0]
 [  5   0]]
Method: Moving Window
Accuracy:  0.9950
Precision: 0.0000
Recall:    0.0000
F1 Score:  0.0000
----------------------------------------
```

The moving window `sample_start` and `sample_end` were adjusted to be 580 and 620, respectively, so that the contextual anomaly would be detected by the moving window method, and ensure that the anomaly would have been missed by a simple threshold method. The following image confirmed this fact:

![Anomaly Detection](./img-2-1-3.png)

![Moving Window](./img-2-1-2.png)
2. **Statistical vs Deep Learning**
**Between ARIMA and LSTM models, which performed better on your dataset and why? Reference specific metrics.**

When comparing ARIMA and LSTM models, the following results are obtained:

| **Metric**   | **ARIMA** | **LSTM**  |
|--------------|-----------|-----------|
| Accuracy     | 0.7750    | 0.9764    |
| Precision    | 0.0947    | 0.6923    |
| Recall       | 0.8214    | 0.3214    |
| F1 Score     | 0.1697    | 0.4390    |

- **Accuracy**, $\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total samples}}$. Accuracy is not reliable in highly imbalanced datasets, which is the case in anomaly detection. A model hence obtain get very high accuracy  if it identifies all points as normal. It has therefore low relevance in anomaly detection.
- **Precision**, $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$. Precision measures the proportion of correctly identified anomalies, hence focusing on false alarms. This is useful when the cost of false positives is high, for example if fraud detection as legitimate transactions can result in cards being blocked and a costly process of investigation. On the other hand, precision is not a good metric for systems where the cost of a false call is negligible when compared to the impact of missing the actual event, such as in earthquake detection. In these cases, it is better to raise false alarms if doing so increases the chances of detecting a real disaster.
- **Recall**, $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$. Recall measures the proportion of actual anomalies that were correctly identified, thus focusing on missed calls. It is essential in scenarios where missing an anomaly carries a high cost. For example, in earthquake detection, a missed call could lead to loss of life. However, it can be misleading in systems where false positives are costly, such as fraud detection. Optimizing for high recall might mean flagging many legitimate transactions as fraudulent, leading to blocked cards and costly investigations.
- **F1 Score**, $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$. F1 balances precision and recall, giving a single summary metric.It is particularly useful the desire to avoid too many false alarms (precision) and the desire not to miss too many anomalies (recall) are balanced.

So in our case:

- ARIMA was better at capturing all anomalies (recall  of 0.82 vs 0.32), but most were incorrect indicating overfitting.
-  LSTM reduced false positives (precision of 0.69 vs 0.09), making its detections more trustworthy.
- LSTM’s balance between precision and recall was far better (F1 of 0.4390 vs 0.1697).

3. **Vanishing Gradient Analysis**
**Explain in your own words how the vanishing gradient problem affects RNNs' ability to detect long-term anomaly patterns, and how LSTMs address this.**

The vanishing gradient problem is a fundamental issue in training RNNs, and it directly impacts their ability to detect long-term anomaly patterns in time series data.
The model computes gradients for each step when using backpropagation through time. These gradients are multiplied repeatedly as they are propagated backward across time steps. If the activation function (like tanh) has derivatives ≤ 1 (as it usually does), this repeated multiplication causes gradients to shrink exponentially. As a result, earlier layers receive gradients close to zero, meaning they learn little or nothing.

In anomaly detection, some patterns may depend on events many steps earlier (e.g., gradual drifts, periodic surges). RNNs with vanishing gradients, therefore, fail to capture these long-range dependencies, making them blind to contextual or collective anomalies that span longer intervals.

LSTMs solve this using:

- **Cell State**: A long-term memory component that flows through time largely unaffected by gradient shrinkage.
Gates:
    - **Forget Gate**: Decides what to discard.
    - **Input Gate**: Decides what to add.
    - **Output Gate**: Controls what to expose to the next layer.

This structure allows gradients to flow through many time steps without vanishing, maintaining a steady signal during training.

Additional references:
- [[https://www.youtube.com/watch?v=AsNTP8Kwu80]]
- [[https://www.youtube.com/watch?v=YCzL96nL7j0]]


4. **LSTM Gate Importance**
**Which of the three gates in an LSTM cell (forget, input, output) do you think is most critical for anomaly detection? Justify your answer.**

The forget gate is the most important component, as it determines which historical information to retain or eliminate from the cell state. In the context of anomaly detection, the model needs to maintain long-term trends to grasp what is deemed "normal" over time. If the forget gate fails to keep pertinent past information, the model risks losing essential temporal context, which may lead to overlooking subtle or delayed anomalies.

If not managed properly, the LSTM might discard important patterns too rapidly (resulting in lost information) or retain them for an extended period (causing overfitting).

The proper functioning of the forget gate is vital in preventing the model from overreacting to standard variations, which can create false positives. This phase aids in identifying issues based on actual differences from significant context that has been preserved.

5. **Autoencoder Threshold Selection**
How did changing the reconstruction error threshold affect precision and recall in your LSTM Autoencoder results? What threshold would you recommend for a production system, and why?

6. **Comparative Analysis**
**For your dataset, rank the three methods (Moving Window, ARIMA, LSTM Autoencoder) by F1-score. Identify one domain where the lowest-performing method might actually be preferable.**

| Rank | Method            | F1 Score |
|------|-------------------|----------|
| 1    | LSTM              | 0.4390   |
| 2    | Feedforward NN    | 0.4000   |
| 3    | LSTM Autoencoder  | 0.1836   |

Autoencoders are a category of neural networks designed to replicate their input. When an autoencoder is trained with typical data, it becomes adept at reproducing standard patterns. When presented with outlier data, the reconstruction error rises, yielding a natural score for anomalies.

LSTM Autoencoders merge the sequence processing strengths of LSTMs with the unsupervised learning of autoencoders, making them suited for detecting anomalies in time series data.

In healthcare settings, particularly in intensive care units, patient vital signs are constantly monitored through sensors that monitor metrics such as pulse rate, blood pressure levels, and the oxygen saturation in the blood (SpO₂).

The collected data is time-series based, and potential anomalies include irregular heartbeats, sudden decreases in oxygen levels, and fluctuations in blood pressure.

LSTM Autoencoders may be effective for medical monitoring due to several factors. 

First, they utilize unsupervised learning, enabling them to identify anomalies in real-time data streams without prior labeling, which is essential since anomalies often go unrecorded. 

Additionally, these models excel at capturing temporal patterns, allowing them to analyze how vital signs fluctuate over time and detect irregularities across multiple time steps. This capability is crucial, as the presentation of anomalies can differ from patient to patient. 

Furthermore, LSTM Autoencoders have the ability to generalize to novel events, recognizing deviations even when they haven't encountered a specific scenario before. 

---

## Notebook 02 – Graph-Based Anomaly Detection

7. **Graph Structure Intuition**
Describe a real-world scenario where modeling data as a graph would reveal anomalies that traditional tabular methods would miss.
8. **Centrality Measure Effectiveness**
Which graph centrality measure (degree, betweenness, clustering coefficient) was most effective at detecting the injected anomalies in your financial transaction network? Explain why.
9. **Node2Vec Parameter Tuning**
How did adjusting the Node2Vec parameters p and q affect the embedding space? Which configuration best separated normal and anomalous nodes?
10. **Community Detection Insight**
When using community-based anomaly detection, what pattern of false positives or false negatives did you observe? What does this suggest about the structure of the anomalies?
11. **GNN Architecture Decision**
If you were to adapt the GCN autoencoder model to detect account takeovers in your transaction network, what specific modifications to the architecture would you make and why?
12. **Comparative Analysis**
Between statistical methods and graph neural networks, which approach showed better performance on structural anomalies versus attribute-based anomalies? Support your answer with metrics from your runs.

---

### Final reflection

13. **Cross-Domain Transfer**
Identify one technique from time series anomaly detection that could be adapted to improve graph-based approaches, or vice versa. Explain how this transfer would work.
14. **Practical Deployment**
If you were to deploy one of these models in a production environment, what three key monitoring metrics would you track over time to ensure continued detection quality?
15. **Key Takeaway**
What was the most surprising or valuable insight you gained from implementing these anomaly detection techniques? (≤3 sentences)

<div style="text-align: center">⁂</div>

[^1]: Reflection_Sheet_AD_Workshop-2.md

