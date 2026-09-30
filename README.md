# COVID-19 MLOps Project

## Project Overview

This project demonstrates an end-to-end MLOps workflow for predicting COVID-19 new cases using a country-wise COVID-19 dataset.

Multiple regression models were experimented with and evaluated. Based on the evaluation results, Ridge Regression was selected as the final model for the MLOps workflow.

The project covers data preprocessing, model experimentation, model training, evaluation, MLflow experiment tracking, reproducibility validation, artifact generation, model registration, and production lifecycle management.

## Objective

The main objective of this project is to predict the number of new COVID-19 cases using country-wise COVID-19 information.

**Target Variable:** `New_cases`

## Dataset

The project uses a country-wise COVID-19 dataset containing:

- 187 records
- 15 columns
- COVID-19 case, death, recovery, and weekly change information
- Country and WHO region information

### Target Variable

`New_cases`

### Features

- `Confirmed`
- `Deaths`
- `Recovered`
- `Active`
- `New_deaths`
- `New_recovered`
- `Deaths_/_100_Cases`
- `Recovered_/_100_Cases`
- `Deaths_/_100_Recovered`
- `Confirmed_last_week`
- `1_week_change`
- `1_week_%_increase`
- `Country/Region`
- `WHO_Region`

## Data Preprocessing

The preprocessing workflow includes:

- Handling infinite values
- Missing-value handling
- Median imputation for numerical features
- Most-frequent imputation for categorical features
- One-hot encoding of categorical features
- StandardScaler for numerical features
- Train-test split with `random_state=42`

### Dataset Split

- Training samples: 149
- Testing samples: 38
- Processed features: 167

## Model Experimentation

Multiple regression models were experimented with during model development.

After comparing the evaluation results, Ridge Regression was selected as the final model used in the MLOps workflow.

### Final Model

**Ridge Regression**

Configuration:

- Alpha: `1.0`
- Random State: `42`

Ridge Regression applies L2 regularization to help control model complexity and reduce the impact of large coefficients.

## Model Evaluation

The final model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Evaluation Results

| Metric | Value |
|---|---:|
| MAE | 745.9113 |
| MSE | 2465046.3020 |
| RMSE | 1570.0466 |
| R² | -0.1748 |

The negative R² value indicates that, on this test split, the model performed worse than a constant mean-prediction baseline in terms of squared-error performance.

## MLOps Workflow

The complete workflow is:

Raw COVID-19 Dataset
↓
Data Preprocessing
↓
Train-Test Split
↓
Feature Transformation
↓
Model Experimentation
↓
Ridge Regression Selection
↓
Model Training
↓
Model Evaluation
↓
MLflow Experiment Tracking
↓
Data and Reproducibility Validation
↓
Artifact Generation
↓
Model Registry
↓
Production Model

## MLflow

MLflow is used for experiment tracking and model management.

The following information is tracked:

- Model parameters
- Training and testing information
- Evaluation metrics
- Model artifacts
- Trained model

### MLflow Experiment

`COVID-19-Ridge-Regression`

### Registered Model

`COVID_Ridge_Model`

The selected model is registered and managed through the MLflow Model Registry.

## Validation

The project includes validation for:

- Raw dataset quality
- Processed data
- Train-test split reproducibility
- Expected feature dimensions
- Model outputs
- Evaluation metrics
- Pipeline outputs

The train-test split uses `random_state=42` to support reproducibility.

## Generated Artifacts

The project generates regression-specific artifacts including:

- `actual_vs_predicted.png`
- `residual_plot.png`
- `prediction_error_distribution.png`
- `model_performance_report.json`
- `evaluation_metrics.json`
- `mlflow_evaluation_metrics.json`
- `registry_report.json`

These artifacts provide evidence of model performance and the MLOps workflow.

