# End-to-end-Machine-Learning-Project-with-MLflow
https://cibil-score-prediction-1.onrender.com/

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



