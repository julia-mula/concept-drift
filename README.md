# Concept Drift in Effort Estimation in IT Projects

Project planning is essential in IT, providing the foundation for scheduling, budgeting, and resource allocation.  
A critical part of planning is **effort estimation**—predicting the work required for tasks or projects.  
Accurate estimates reduce risks, while poor ones lead to delays and cost overruns.

Traditional estimation methods (expert judgment, analogy, models like **COCOMO**) and modern approaches (machine learning, data-driven techniques) have improved predictions but still face challenges, especially in **Agile environments**. Studies show that even story point estimation often lacks accuracy across projects.

A major factor is **concept drift**—changes in requirements, technologies, or team practices that reduce model accuracy over time.  

This project investigates:  
- Can concept drift be detected in effort estimation?  
- Can it be countered effectively with existing strategies?

## Results

Experiments were performed on **15 IT project datasets**.

Each project was evaluated across four performance metrics:  
- **MAE** (Mean Absolute Error)  
- **MSE** (Mean Squared Error)  
- **R²** (Coefficient of Determination)  
- **MMRE** (Mean Magnitude of Relative Error)  

The **full results** (all projects and metrics) are available in the [`results/`](./results/) folder.

### Concept Drift Detection

To study the impact of **concept drift**, the **ADWIN** algorithm was applied for drift detection.  
Two scenarios were compared after drift detection:  
- **No Retraining** – models continue without adaptation.  
- **Retraining** – models are retrained after drift is detected.  

Performance metrics were analyzed for both scenarios, showing the benefit (or limitations) of retraining in maintaining estimation accuracy.

