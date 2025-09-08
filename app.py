from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")



@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    
    if request.method == 'POST':
        
        try:
            # reading the inputs given by the user
            
            required_fields = [
                'name', 'age', 'occupation', 'bank', 'number_of_banks', 'number_of_loans',
                'due_loans', 'hard_checks', 'credit_limit', 'credit_usage', 'monthly_income',
                'total_limit', 'debt_to_income_ratio'
            ]
            for field in required_fields:
                if field not in request.form or request.form[field] == '':
                    return render_template('index.html', error=f"Missing or empty field: {field}")

            name = str(request.form['name'])
            try:
                age = int(request.form['age'])
                number_of_banks = int(request.form['number_of_banks'])
                number_of_loans = int(request.form['number_of_loans'])
                due_loans = int(request.form['due_loans'])
                hard_checks = int(request.form['hard_checks'])
                credit_limit = float(request.form['credit_limit'])
                credit_usage = float(request.form['credit_usage'])
                monthly_income = float(request.form['monthly_income']) if 'monthly_income' in request.form else 0.0
                total_limit = float(request.form['total_limit'])
                debt_to_income_ratio = float(request.form['debt_to_income_ratio'])
            except ValueError as ve:
                return render_template('index.html', error="Please enter valid numeric values for numeric fields.")

            occupation = str(request.form['occupation'])
            bank = str(request.form['bank'])

            data = [age, number_of_banks, number_of_loans, due_loans, hard_checks, credit_limit, credit_usage, monthly_income, total_limit, debt_to_income_ratio]
            data = np.array(data).reshape(1, 10)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)
            print(predict)
            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            print('The Exception message is: ', e)
            return render_template('index.html', error="An unexpected error occurred. Please try again.")

    else:
        return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)