# Mama-Tee-Restaurant-Model
## By Adeniyi Aishat
Building a regression task to predict the amount of tip
# MAMA TEE

**Author:** Adeniyi Aishat  

---

## Project Overview

**MAMA TEE** is a simple **Machine Learning Regression Project** built with **Python** and **Scikit-learn**.  
The goal of this project is to predict **tips amount** based on customer dining data such as total bill, gender, smoking status, day, time, and table size.

The dataset used is **`tips.csv`**, which contains 744 rows and 7 columns.

---

## Dataset Description

| Column | Description |
|---------------|-------------|
| `total_bill` | Total bill amount |
| `tip` | Tip amount given |
| `gender` | Gender of the customer |
| `smoker` | Whether the customer is a smoker or not |
| `day` | Day of the week |
| `time` | Time of the meal (Lunch/Dinner) |
| `size` | Size of the table (number of people) |

---

## Libraries Used


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np



## Data Preprocessing

* Checked for missing values (none found).
* One-hot encoded categorical columns (`gender`, `smoker`, `day`, `time`).
* Split dataset into **training (70%)** and **testing (30%)** sets using `train_test_split()`.

---

## Model Building

Three regression models were trained and compared:

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**

Each model was evaluated using **MAE**, **RMSE**, and **R² Score**.

---

## Model Performance

| Model | MAE | RMSE | R² Score |
| ----------------- | ------ | ------ | -------- |
| Linear Regression | 118.90 | 151.22 | 0.0158 |
| Decision Tree | 149.52 | 199.90 | -0.7197 |
| Random Forest | 120.50 | 154.85 | -0.0320 |

---

## Code Snippet Example


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Prediction
y_pred = reg.predict(X_test)

# Evaluation
from sklearn.metrics import mean_absolute_error
print("MAE:", mean_absolute_error(y_test, y_pred))



##  Insights

* The **Linear Regression model** performed slightly better than the Decision Tree and Random Forest in terms of error metrics.
* However, all models have relatively **low R² scores**, indicating that the input variables may not strongly predict the tip amount.

---

## Conclusion

This project demonstrates the basic workflow of:

* Data preprocessing
* Feature encoding
* Model training and evaluation

Future improvements could include:

* Feature engineering (e.g., creating interaction terms)
* Hyperparameter tuning
* Using more complex models such as **XGBoost** or **Gradient Boosting**

---

## Author

**Adeniyi Aishat**
Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/adeniyi-aishat-himisioluwa)
