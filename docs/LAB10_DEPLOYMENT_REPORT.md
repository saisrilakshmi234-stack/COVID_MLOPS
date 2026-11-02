# LAB 10 – Production Model Deployment Report

## 1. Objective
The objective of Lab 10 was to deploy the trained COVID-19 prediction model from the MLflow Model Registry Production stage.

## 2. Model Details
- Project: COVID-19 MLOps
- Model name: COVID_Ridge_Model
- Model version: 6
- Model stage: Production
- Algorithm: Ridge Regression
- Prediction target: New_cases

## 3. Deployment Activities
1. Connected to the MLflow tracking database.
2. Retrieved the Production model from the MLflow Model Registry.
3. Loaded the saved preprocessing pipeline.
4. Validated preprocessing and sample predictions.
5. Exported the trained model to deployment/model.pkl.
6. Exported the preprocessing pipeline to deployment/preprocessor.pkl.
7. Generated deployment metadata in deployment/model_metadata.json.

## 4. Exported Artifacts
- deployment/model.pkl
- deployment/preprocessor.pkl
- deployment/model_metadata.json

## 5. Validation Results
- Model loaded successfully: PASS
- Preprocessor loaded successfully: PASS
- Model and preprocessor compatibility: PASS
- Transformed feature count: 167
- Sample predictions validated: PASS
- Sample prediction count: 5

## 6. Conclusion
The COVID_Ridge_Model version 6 was successfully retrieved from the MLflow Production stage. The model and preprocessing pipeline were exported and validated for API deployment.
