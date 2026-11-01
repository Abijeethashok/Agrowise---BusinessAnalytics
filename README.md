# AgroWise – Business Analytics-Based Crop Recommendation System

AgroWise is a machine-learning and business-analytics web application that recommends suitable crops from seven soil and environmental inputs: nitrogen (N), phosphorus (P), potassium (K), temperature, relative humidity, soil pH, and rainfall. It covers 22 crop categories and presents recommendations, crop assessments, reference profiles, and statistical insights through a Flask web interface.

## Features

- Crop recommendations based on seven user-provided input parameters.
- Comparison of Decision Tree, Gaussian Naive Bayes, and Random Forest models; the saved project selects Gaussian Naive Bayes.
- Crop suitability assessment and data-derived reference profiles.
- Analytics dashboard with descriptive statistics, correlation analysis, outlier summaries, statistical group comparisons, model evaluation, and feature information.
- Flask pages for the home screen, recommendation form, analytics dashboard, and project information.

## Technology stack

Python, Flask, Pandas, NumPy, SciPy, scikit-learn, Statsmodels, Joblib, HTML, CSS, and JavaScript.

## Run locally

The project was developed using Python 3.11.6.

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and start the development server:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser. The health endpoint is available at `http://127.0.0.1:5000/api/health`.

## Deploy on Render

This repository includes a `render.yaml` Blueprint. In Render, choose **New → Blueprint**, connect this GitHub repository, and follow the prompts. The Blueprint installs the dependencies and starts the Flask app with Gunicorn. Render will provide an `onrender.com` URL after a successful deployment.

The deployment uses `/api/health` as its health-check endpoint. The app is configured to bind to Render's assigned `$PORT`.

## Repository structure

```text
.
├── app.py
├── recommender.py
├── clean_data.py
├── inspect_data.py
├── analyze_data.py
├── build_profiles.py
├── train_model.py
├── data/                 # Source and cleaned crop datasets
├── models/               # Trained model and preprocessing artifacts
├── outputs/              # Saved statistical and evaluation outputs
├── templates/            # Flask HTML templates
├── static/               # CSS and JavaScript assets
├── test_*.py            # Project test scripts
├── requirements.txt
├── .python-version
└── render.yaml
```

## Dataset and interpretation notes

The project uses the public *Crop Recommendation dataset* posted on Figshare by Saroj Mali (2024, Version 2): <https://figshare.com/articles/dataset/Crop_Recommendation_dataset/26308696>.

The dataset is described by its publisher as augmented. Model scores indicate similarity to records in this dataset; they are **not** probabilities of successful cultivation or expected yield. The reference profiles are also calculated from this same dataset and should not be treated as independent agronomic validation. In particular, N/P/K are represented in the dataset's own units/ratios, and the source does not specify the rainfall averaging period. Real-world decisions should be checked against local soil tests and qualified agricultural guidance.

## Notes for contributors

- Keep the trained files in `models/`, the datasets in `data/`, and the saved analysis files in `outputs/`; the running application reads these files.
- Do not commit virtual environments, local `.env` files, credentials, or API keys.
- When changing the model or its preprocessing, retrain and validate the model artifacts together before deployment.
