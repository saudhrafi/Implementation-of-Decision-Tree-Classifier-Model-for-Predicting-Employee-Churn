# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Read the Employee.csv dataset and check its structure and missing values.

2.Convert the categorical salary column into numeric values using LabelEncoder.

3.Select important features like satisfaction_level, number_project, salary, etc., as input (X) and take left as output (Y).

4.Split the dataset into training (80%) and testing (20%) sets.

5.Train a Decision Tree classifier using entropy as the splitting criterion.

6.Predict employee attrition on test data and calculate the model accuracy.

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Raha Priya Dharshini M
RegisterNumber:  212224240124
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv("Employee.csv")

data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()

le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

data.head()

x = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary"
]]

x.head()

y = data["left"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)

dt = DecisionTreeClassifier(criterion="entropy")

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy

dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])

plt.figure(figsize=(8, 6))
plot_tree(
    dt,
    feature_names=x.columns,
    class_names=["stayed", "left"],
    filled=True
)
plt.show()

```

## Output:

<img width="1375" height="439" alt="Screenshot 2026-02-13 105017" src="https://github.com/user-attachments/assets/472686d5-a07f-4b37-b7c4-3b31e330cef0" />

<img width="1383" height="603" alt="image" src="https://github.com/user-attachments/assets/eba8ad34-f086-46d1-b7e1-f97a7a0103ae" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
