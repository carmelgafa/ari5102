
# Anomaly Detection Workshop – Reflection Sheet

**Name(s):** ______________________________________  **Date:** ____________

Instructions  
*Work individually or in pairs. Answer concisely (bullet points are fine). Save this file (or export to PDF) and email it to Charlie by **18:00 CET tomorrow**. Feel free to reference code, figures, or metrics from your notebook runs.*

---

## Notebook 01 – Statistical Foundations

1. **Conceptual distinction**  
   Give one real‑world example of a *contextual* anomaly and explain why a univariate Z‑score could miss it.

2. **Robust Z‑score metric**  
   On your run, what PR‑AUC did the Robust Z‑score achieve? Briefly interpret its curve shape (early precision vs recall).

3. **Mahalanobis insight**  
   Did Mahalanobis distance outperform Z‑score for this dataset? If yes, what property of the data explains the gain? If no, why might Mahalanobis have struggled?

4. **Threshold sensitivity**  
   Predict the effect of lowering the Z‑score threshold from the 99.9‑th percentile to the 99.5‑th. Which error types rise?

---

## Notebook 02 – Unsupervised Methods

5. **Model ranking**  
   Rank Isolation Forest, One‑Class SVM, and Autoencoder by PR‑AUC on your run. State one strength that justifies the top model’s performance.

6. **Hyper‑parameter impact**  
   Choose a hyper‑parameter you tuned (e.g., `contamination`, `ν`, latent dimension). Describe how changing it affected the Precision‑Recall curve.

7. **Metric selection**  
   In a domain where false negatives are costlier than false positives, which evaluation metric would you optimise during tuning and why?

8. **Method suitability**  
   Name a scenario where OC‑SVM might beat Isolation Forest, and justify based on data characteristics.

---

## Notebook 03 – Explainability

9. **IF SHAP reading**  
   In your per‑instance SHAP force plot, which feature had the largest red contribution and what actionable insight does that provide?

10. **AE SHAP caveat**  
    Why can SHAP explanations of raw reconstruction error become noisy in very high‑dimensional data, and how could you mitigate this?

11. **Trust & regulation**  
    Draft a one‑sentence, GDPR‑compliant explanation you could provide to a user whose transaction was flagged anomalous by the IF model.

12. **Model debugging**  
    Describe one way SHAP helped you spot a potential model weakness or data issue during the workshop.

---

### Final reflection

13. **Key takeaway**  
    What was the most surprising or valuable thing you learned across all three notebooks? (≤3 sentences)
