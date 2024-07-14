# Diabetes-Detection Using Flask

## Overview
This project utilizes machine learning random forest algorithm to predict diabetes outcomes based on patient data.
It features a user-friendly web interface built with Flask, allowing users to input their health metrics and receive instant predictions.

### Supervised by 
[Prof. Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu) ,
(Assistant Professor) Department of CSE, MIT Mysore

### Collaborators
4MH21CS030 [Hajeera Suhani](https://github.com/hajira25)

4MH21CS052 [Mohammed Abid IS](https://github.com/mdabid15)

4MH21CS053 [Mohammed Ibrahim Khan](https://github.com/ibrahim2604)

4MH21CS102 [Sufiya Salam](https://github.com/sufiyasalam)

## Website

[Diabetes-Detection](http://127.0.0.1:5000/)

## Project Structure and Description 

```
Diabetes Detection/
│
├── app.py                          # Main application file
├── train_model.py                  # Script to train the model
├── dataset/
│   └── diabetes.csv                # Dataset for training
├── templates/
│   ├── index.html                  # Input form
│   └── results.html                # Results display
├── models/
│   ├── diabetes_detection_model.h5  # Pre-trained model .h5 extension
│   └── rf_model.pkl                # Random Forest model
└── requirements.txt                # Dependencies
```

## Prerequisites

- Python 3.12
- Flask
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- TensorFlow (if using .h5 model)
- HTML/CSS for frontend design

## Installation

You can install the required packages using:

```bash
pip install -r requirements.txt
```
## Setup Instructions

### Step 1: Clone the Repository

Clone the repository to your local machine:

```bash
git clone <repository_url>
cd Diabetes Detection
```

### Step 2: Set Up a Virtual Environment (Optional but Recommended)

Create and activate a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

Run the Flask application:

```bash
python app.py
```

Open your web browser and navigate to `http://127.0.0.1:5000/` to access the application.

## How to Use the Application

1. Input your health data in the provided fields.
2. Click on the "Submit" button to see the results.
3. The application will display whether you are diabetic or not along with relevant metrics.

## Training the Model

To train the model from scratch, run the `train_model.py` script:

```bash
python train_model.py
```

This will generate the model files (`diabetes_detection_model.h5` and `rf_model.pkl`) used by the application.

## Screenshot
![Screenshot 2024-07-14 154721](https://github.com/user-attachments/assets/c6f8f55e-943c-452e-bde6-86e089390e15)

# Diabetes Detection Using Streamlit

## Overview
This project utilizes machine learning random forest algorithm to predict diabetes outcomes based on patient data.
It features a user-friendly web interface built with Streamlit Community Cloud, allowing users to input their health metrics and receive instant visualized predictions.

### Supervised by 
[Prof. Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu) ,
(Assistant Professor) Department of CSE, MIT Mysore

## Website

[Diabetes-Detection](https://diabetesdetection-ss.streamlit.app/)

## Project Structure and Description  
- Remains same as above.

## Prerequisites

- Python 3.12
- Streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- TensorFlow (if using .h5 model)
- HTML/CSS for frontend design

## Installation
- Refer the above till Step 3

### Step 4: Run the Application
1. **Ensure you're in the project directory.**

2. **Run the Streamlit app:**

   ```bash
   streamlit run app.py

## Usage

*   Input patient data using the sliders in the sidebar.
*   Click the "Submit" button to get predictions and visualizations.
*   The app will display the patient's data along with the prediction and accuracy of the model.

## Screenrecording



https://github.com/user-attachments/assets/75b69a8a-28a4-42c2-9237-154f97a9a9a0




## Conclusion

The Diabetes Detection application leverages machine learning to provide insights into an individual's diabetes risk based on their health data. By using a Random Forest Classifier, the model can accurately predict outcomes, empowering users to make informed health decisions. This tool not only aids in early detection but also promotes awareness and proactive health management.


