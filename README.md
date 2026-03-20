# 💻 Laptop Price Prediction (End-to-End Machine Learning Project)

##  Overview

This project is an **end-to-end machine learning pipeline** designed to predict laptop prices based on various hardware and software features. The goal is to simulate a real-world ML system, covering everything from data ingestion to model deployment readiness.

The project demonstrates practical skills in:

* Data preprocessing
* Feature engineering
* Model training and evaluation
* Pipeline structuring
* Error handling and logging

---

## Project Workflow

The project follows a modular pipeline architecture:

1. **Data Ingestion**

   * Reads raw dataset
   * Splits into training and testing data
   * Stores processed datasets

2. **Data Transformation**

   * Handles missing values
   * Encodes categorical variables (One-Hot Encoding)
   * Scales numerical features (StandardScaler)
   * Uses `ColumnTransformer` for pipeline consistency
   * Saves preprocessing object for reuse

3. **Model Training**

   * Trains multiple regression models:

     * Random Forest
     * Decision Tree
     * Gradient Boosting
     * Linear Regression
     * AdaBoost
   * Performs hyperparameter tuning using GridSearchCV
   * Selects the best model based on R² score

4. **Model Evaluation**

   * Evaluates performance on test data
   * Uses R² score as primary metric

5. **Model Saving**

   * Saves trained model and preprocessing pipeline using `dill`

---

## Features Used

### Numerical Features

* Inches
* RAM
* Weight
* Screen Width
* Screen Height

### Categorical Features

* Company
* TypeName
* CPU
* Memory
* Operating System

---

##  Models Implemented

| Model             | Description                 |
| ----------------- | --------------------------- |
| Random Forest     | Ensemble learning (bagging) |
| Decision Tree     | Tree-based model            |
| Gradient Boosting | Boosting-based ensemble     |
| Linear Regression | Baseline linear model       |
| AdaBoost          | Adaptive boosting model     |

---

##  Project Structure

```
mlproject/
│
├── artifact/                 # Saved models and preprocessors
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebooks/               # Jupyter notebooks for EDA & experiments
│
├── src/
│   ├── components/
│   │   ├── dataingestion.py
│   │   ├── datatransformer.py
│   │   └── model_trainer.py
│   │
│   ├── utils.py             # Utility functions (save_object, evaluation)
│   ├── logger.py            # Logging configuration
│   └── exception.py         # Custom exception handling
│
├── logs/                    # Log files
├── requirements.txt
└── README.md
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/laptop-price-prediction.git
cd laptop-price-prediction
```

2. Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the pipeline from the root directory:

```bash
python -m src.components.dataingestion
```

This will:

* Load data
* Transform features
* Train models
* Save the best model

---

## Evaluation Metric

* **R² Score (Coefficient of Determination)**
  Used to measure how well the model explains variance in laptop prices.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Dill (for model serialization)

---

## Key Learnings

* Building modular ML pipelines
* Handling categorical and numerical data efficiently
* Debugging real-world ML issues (e.g., sparse matrices, model fitting errors)
* Hyperparameter tuning and model selection
* Writing production-ready code with logging and exception handling

---

##  Future Improvements

* Add Flask/FastAPI deployment
* Integrate with a frontend UI
* Deploy to cloud (AWS / Azure)

---



## 📬 Contact

**Zukisa Mkhize**
Applied Mathematics Graduate | Aspiring Data Scientist

* LinkedIn: https://www.linkedin.com/in/zukisa-mkhize-0b56351b8/
* GitHub:(https://github.com/Zukisa313)

---

## If you like this project

Give it a star on GitHub!
