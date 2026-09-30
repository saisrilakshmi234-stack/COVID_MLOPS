\# COVID-19 MLOps Project



\## Project Overview



This project demonstrates an end-to-end MLOps workflow for predicting COVID-19 new cases using a country-wise COVID-19 dataset.



Multiple regression models were experimented with and evaluated, and Ridge Regression was selected as the final model based on the evaluation results. The selected model was then integrated into an MLOps workflow using preprocessing pipelines, MLflow experiment tracking, model validation, model registry, and production lifecycle management.



\## Objective



The main objective of this project is to build a reproducible and manageable machine learning workflow for predicting:



\*\*Target Variable:\*\* `New\_cases`



The project covers the complete workflow from data preprocessing and model training to experiment tracking, validation, artifact generation, and model registration.



\## Dataset



The project uses a country-wise COVID-19 dataset containing:



\- 187 records

\- 15 columns

\- COVID-19 case, death, recovery, and weekly change information

\- Country and WHO region information



\### Target



`New\_cases`



\### Important Features



\- Confirmed

\- Deaths

\- Recovered

\- Active

\- New\_deaths

\- New\_recovered

\- Deaths\_/\_100\_Cases

\- Recovered\_/\_100\_Cases

\- Deaths\_/\_100\_Recovered

\- Confirmed\_last\_week

\- 1\_week\_change

\- 1\_week\_%\_increase

\- Country/Region

\- WHO\_Region



\## Data Preprocessing



The preprocessing workflow includes:



\- Handling infinite values

\- Missing-value handling

\- Numerical feature median imputation

\- Categorical feature most-frequent imputation

\- One-hot encoding of categorical features

\- StandardScaler for numerical features

\- Train-test split with `random\_state=42`

\- `80%` training data and `20%` test data



\### Processed Dataset



\- Training samples: 149

\- Test samples: 38

\- Processed features: 167



\## Model Experimentation



Multiple regression models were experimented with during model development.



After comparing the model evaluation results, \*\*Ridge Regression\*\* was selected as the final model used in the MLOps workflow.



\### Final Model



\*\*Ridge Regression\*\*



Configuration:



\- Alpha: `1.0`

\- Random State: `42`



Ridge Regression uses L2 regularization, which helps control model complexity and reduce the effect of large coefficients.



\## Model Evaluation



The final Ridge Regression model was evaluated using:



\- Mean Absolute Error (MAE)

\- Mean Squared Error (MSE)

\- Root Mean Squared Error (RMSE)

\- R² Score



\### Final Evaluation Results



| Metric | Value |

|---|---:|

| MAE | 745.9113 |

| MSE | 2465046.3020 |

| RMSE | 1570.0466 |

| R² | -0.1748 |



The negative R² value indicates that, on this test split, the model performed worse than a constant mean-prediction baseline in terms of squared-error performance.



\## MLOps Workflow



The project is organized as an end-to-end MLOps workflow:



```text

Raw COVID-19 Dataset

&#x20;       ↓

Data Preprocessing

&#x20;       ↓

Train-Test Split

&#x20;       ↓

Feature Transformation

&#x20;       ↓

Model Experimentation

&#x20;       ↓

Ridge Regression Selection

&#x20;       ↓

Model Training

&#x20;       ↓

Model Evaluation

&#x20;       ↓

MLflow Experiment Tracking

&#x20;       ↓

Data \& Reproducibility Validation

&#x20;       ↓

Artifact Generation

&#x20;       ↓

Model Registry

&#x20;       ↓

Production Model

