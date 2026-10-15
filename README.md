# MachinaSense
### Machine Learning-Based Predictive Maintenance Dashboard

MachinaSense is a machine learning-powered predictive maintenance application designed to identify potential industrial machine failures using sensor data. It combines machine learning, anomaly detection, and an interactive web dashboard to help users analyze machine conditions, identify potential failures, and prioritize maintenance activities.

The project uses the AI4I 2020 Predictive Maintenance Dataset to train and evaluate a machine failure prediction model. Users can explore sensor data, view model evaluation metrics, analyze uploaded CSV files, and review maintenance alerts with recommended actions.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Machine Learning Model](#machine-learning-model)
- [Model Evaluation](#model-evaluation)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Testing and Validation](#testing-and-validation)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Overview

Industrial equipment failures can cause production delays, unexpected maintenance costs, and operational disruptions. Traditional maintenance approaches may rely on fixed schedules or react only after a failure occurs.

MachinaSense explores a data-driven alternative by using machine sensor measurements to estimate the likelihood of machine failure. The application presents predictions and model metrics through a web-based dashboard, making the results easier to inspect and interpret.

The system is intended as a prototype for predictive maintenance workflows and educational experimentation. It is not a replacement for industrial safety systems or professional maintenance assessments.

## Problem Statement

Develop a machine learning-based predictive maintenance system that analyzes industrial sensor data, identifies potential machine failures, and presents actionable maintenance information through an interactive web dashboard.

## Objectives

- Develop a machine learning model for industrial machine failure prediction.
- Analyze sensor data and identify potential failure conditions.
- Present model performance through evaluation metrics.
- Support CSV uploads for batch prediction.
- Display predicted failures and maintenance recommendations.
- Validate input data and handle invalid requests reliably.
- Provide a modular application that can be extended for future industrial use cases.

## Key Features

### 1. Predictive Failure Detection

Uses a trained Random Forest classification model to estimate machine failure risk from sensor measurements.

### 2. Anomaly Detection

Includes an Isolation Forest-based anomaly detection component for identifying potentially unusual sensor patterns.

### 3. Interactive Dashboard

Displays model evaluation information and sensor data visualizations through a web interface.

### 4. Sensor Data Visualization

Provides a sensor selection control that allows users to explore different sensor measurements.

### 5. CSV-Based Batch Prediction

Accepts CSV files for batch analysis and returns prediction results for the submitted records.

### 6. Maintenance Alerts

Displays predicted failure alerts with severity classifications and recommended maintenance actions.

### 7. Input Validation

Checks incoming sensor data and rejects invalid or incomplete inputs through the application's API.

### 8. Model Metrics and Health Endpoints

Provides API endpoints for application health checks and model evaluation information.

## System Architecture

The application follows a modular architecture consisting of data processing, model training, prediction, API handling, and dashboard presentation.

```text
Industrial Sensor Dataset / CSV Upload
                 |
                 v
        Data Validation
                 |
                 v
       Feature Processing
                 |
                 v
      Trained ML Pipeline
       /             \
      v               v
Failure Prediction   Anomaly Detection
       \             /
        v           v
       Prediction Results
                 |
                 v
          Flask REST API
                 |
                 v
       Web-Based Dashboard
                 |
                 v
       Alerts and Metrics
```

The Random Forest model performs failure classification, while the Isolation Forest component supports anomaly detection. The Flask backend exposes application functionality to the frontend, which renders the results and visualizations.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend Framework | Flask |
| Machine Learning | scikit-learn |
| Classification Model | Random Forest |
| Anomaly Detection | Isolation Forest |
| Data Processing | pandas, NumPy |
| Model Serialization | joblib |
| Frontend | HTML, CSS, JavaScript |
| Dataset | AI4I 2020 Predictive Maintenance Dataset |
| Version Control | Git and GitHub |

## Machine Learning Model

MachinaSense uses the AI4I 2020 Predictive Maintenance Dataset, which contains 10,000 records representing machine operating conditions and failure information.

The dataset contains 9,661 normal records and 339 failure records, giving a failure prevalence of approximately 3.39%.

### Model Development Workflow

1. Load and validate the dataset.
2. Separate the data into training, validation, and test sets.
3. Train the Random Forest classification pipeline.
4. Select a classification threshold using validation data.
5. Evaluate the model using held-out test data.
6. Save the trained model for application inference.
7. Use the trained pipeline to generate predictions through the Flask application.

### Data Split

| Dataset | Records |
|---|---:|
| Training | 7,000 |
| Validation | 1,500 |
| Testing | 1,500 |
| Total | 10,000 |

The reported test metrics are calculated on the held-out test set. The classification threshold selected for the current model is 0.35.

## Model Evaluation

The current model achieved the following results on the held-out test set.

| Metric | Result |
|---|---:|
| Accuracy | 98.80% |
| Precision | 70.37% |
| Recall | 65.52% |
| F1-score | 67.86% |
| Average Precision | 0.6849 |
| False Positives | 8 |
| False Negatives | 10 |
| True Positives | 19 |
| True Negatives | 1,463 |

### Confusion Matrix

| Actual / Predicted | Normal | Failure |
|---|---:|---:|
| Normal | 1,463 | 8 |
| Failure | 10 | 19 |

### Interpretation

- **Accuracy:** 98.80% of the test records were classified correctly.
- **Precision:** Approximately 70.37% of predicted failures were actual failures.
- **Recall:** Approximately 65.52% of actual failures were identified by the model.
- **F1-score:** 67.86%, balancing precision and recall.
- **Average Precision:** 0.6849, measuring precision-recall performance across classification thresholds.

Because machine failures are relatively uncommon in the dataset, accuracy alone does not adequately describe model performance. Precision, recall, F1-score, and the confusion matrix provide additional information about missed failures and false alarms.

These results represent performance on the specified dataset and test split. They should not be interpreted as guaranteed performance on real industrial machinery.

## Project Structure

```text
MachinaSense/
|
|-- app.py
|-- train_model.py
|-- test_robustness.py
|-- requirements.txt
|-- .gitignore
|
|-- data/
|   |-- ai4i_dataset.csv
|   |-- upload_test.csv
|   |-- invalid_test.csv
|   |-- missing_values_test.csv
|
|-- models/
|   |-- machinasense.joblib
|
|-- reports/
|   |-- metrics.json
|
|-- static/
|   |-- css/
|   |   |-- style.css
|   |
|   |-- js/
|       |-- script.js
|
|-- templates/
|   |-- index.html
|
|-- README.md
```

Generated model files and selected backup or prediction files are excluded from version control. The model is generated locally by the training script.

## Installation and Setup

### Prerequisites

Install the following before running the project:

- Python 3.12
- Git
- A terminal or code editor
- Internet access for installing Python dependencies

### 1. Clone the Repository

```bash
git clone https://github.com/pawartarun3228/MachinaSense.git
cd MachinaSense
```

### 2. Create a Virtual Environment

On Windows using Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Alternatively, use PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the Model

The trained model files are excluded from the Git repository. Generate them locally using the training script:

```bash
python train_model.py
```

Ensure that the dataset is available at the path expected by the training script and that the script completes successfully.

## Running the Application

After installing dependencies and generating the model artifacts, start the Flask application:

```bash
python app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

The dashboard can then be used to inspect available metrics, explore sensor visualizations, upload CSV data, and review generated predictions and alerts.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Checks application health and readiness information |
| `/api/metrics` | GET | Returns model evaluation metrics and dataset split information |
| `/api/alerts` | GET | Returns maintenance alerts generated from prediction results |

The application also accepts sensor input and CSV uploads through its implemented prediction routes. Refer to `app.py` for the exact request formats, required feature names, and response structures.

## Testing and Validation

MachinaSense includes a robustness testing script:

```bash
python test_robustness.py
```

The recorded test run passed all seven checks in the project's robustness suite, covering:

- Backend health and required model/data artifacts.
- Valid sensor input.
- Missing required features.
- Invalid numeric input.
- Empty numeric values.
- Small perturbations to sensor input.

CSV upload behavior has also been checked using valid, invalid, and missing-value test files.

These results describe the recorded test run; rerun the tests in your current environment to verify the latest code.

## Limitations

- The model is trained and evaluated using a benchmark dataset rather than live industrial equipment.
- Real-world performance may differ because of changes in sensors, machine types, operating conditions, and failure distributions.
- The dataset contains relatively few failure records
