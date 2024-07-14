from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)

# Load data
cwd = os.getcwd()
file_path = os.path.join(cwd, 'diabetes.csv')
df = pd.read_csv(file_path)

# Prepare data
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Save the model
model_path = os.path.join(cwd, 'rf_model.pkl')
joblib.dump(rf, model_path)

# Load model
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    user_data = np.array([
        float(data['Pregnancies']),
        float(data['Glucose']),
        float(data['BloodPressure']),
        float(data['SkinThickness']),
        float(data['Insulin']),
        float(data['BMI']),
        float(data['DiabetesPedigreeFunction']),
        float(data['Age'])
    ]).reshape(1, -1)

    prediction = model.predict(user_data)
    output = 'You are Diabetic' if prediction[0] == 1 else 'You are not Diabetic'
    accuracy = accuracy_score(y_test, model.predict(x_test)) * 100

    return render_template('result.html', output=output, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
