
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

The moving window `sample_start` and `sample_end` were adjusted to be 580 and 620, respectively, so that the contextual anomaly would be detected by the moving window method, and ensure that the anomaly would have been missed by a simple threshold method. The following image confirmed this fact:

![Moving Window](./img-2-1-2.png)


2. **Statistical vs Deep Learning**
Between ARIMA and LSTM models, which performed better on your dataset and why? Reference specific metrics.
3. **Vanishing Gradient Analysis**
Explain in your own words how the vanishing gradient problem affects RNNs' ability to detect long-term anomaly patterns, and how LSTMs address this.
4. **LSTM Gate Importance**
Which of the three gates in an LSTM cell (forget, input, output) do you think is most critical for anomaly detection? Justify your answer.
5. **Autoencoder Threshold Selection**
How did changing the reconstruction error threshold affect precision and recall in your LSTM Autoencoder results? What threshold would you recommend for a production system, and why?
6. **Comparative Analysis**
For your dataset, rank the three methods (Moving Window, ARIMA, LSTM Autoencoder) by F1-score. Identify one domain where the lowest-performing method might actually be preferable.

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

