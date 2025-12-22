# CIBIL Score Prediction — End-to-End ML Project

Live demo: https://cibil-score-prediction-1.onrender.com/

A small end-to-end machine learning project that trains and serves a model to predict a customer's CIBIL score category. This repo contains data ingestion, transformation, model training, MLflow tracking, and a Flask web app demo.

## Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Tech stack](#tech-stack)
- [Quick start](#quick-start)
- [How to use the app](#how-to-use-the-app)
- [Model & results](#model--results)
- [CI / Tests](#ci--tests)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains an end-to-end ML pipeline and a simple web UI to try predictions. It is organized so that the pipeline can be run programmatically (via `main.py`) and the web demo is available via `app.py`.

## Demo

Live demo: https://cibil-score-prediction-1.onrender.com/

Include screenshots here (drag & drop or add to `static/` and reference them).

## Tech stack
- Python 3.8+
- Flask (web UI)
- scikit-learn / XGBoost (models)
- MLflow (experiment tracking)
- Docker (optional container)

## Quick start

1. Create a virtual environment and activate it:

```bash
python -m venv my_env
my_env\Scripts\activate    # Windows
# or on macOS / Linux: source my_env/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the web app (development):

```bash
python app.py
# Open http://localhost:8080
```

4. Run the pipeline / training:

```bash
python main.py
```

### Docker (optional)

Build and run the Docker image (you have a `Dockerfile`):

```bash
docker build -t cibil-score-app .
docker run -p 8080:8080 cibil-score-app
```

## How to use the app

- Open the demo link or `http://localhost:8080`.
- Fill the form with the requested features and click Predict.
- The result page shows the predicted score category.

If you prefer API calls, POST to `/predict` with form-encoded fields (the web form names are used as keys).

## Model & results

Model files and artifacts are stored in `artifacts/` after training. Add model evaluation metrics and visualizations here (accuracy, confusion matrix, sample predictions). Consider adding CSVs or images into `static/` and linking them here.

## CI & Tests

This repo includes a minimal GitHub Actions workflow (`.github/workflows/ci.yml`) to run tests. Tests are in `tests/` and use `pytest`.

## Contributing

Contributions are welcome! See `CONTRIBUTING.md` for the process. Create issues for bugs or feature requests and open a pull request with a clear description.

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
=======

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py



# How to run?
### STEPS:

Clone the repository

### STEP 01- Create a conda environment after opening the repository

```bash
python -m venv my_env
```

```bash
my_env\Scripts\activate
```




### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/narraranjith22/cibil_score_prediction.mlflow
MLFLOW_TRACKING_USERNAME=NarraRanjith \
MLFLOW_TRACKING_PASSWORD=password \
python script.py



