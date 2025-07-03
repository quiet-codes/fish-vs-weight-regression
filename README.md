# fish-vs-weight-regression
This project applies Univariate Linear Regression to predict the weight of a fish based on its length (specifically Length2) from the Fish Market dataset.

🔍 Objective
The goal is to:

1.Understand the relationship between fish length and weight

2.Implement linear regression from scratch using gradient descent

3.Learn how the cost function and gradient function help the model improve

4.Predict weights for unseen lengths using the learned parameters

📊 Dataset Used
Dataset: Fish.csv

Feature (x): Length2 (a numeric measure of the fish's length)

Target (y): Weight (grams)

🧠 Algorithm Used
This project uses Linear Regression with Gradient Descent to minimize the Mean Squared Error (MSE) between predicted and actual weights.

Hypothesis Function:

ŷ = w * x + b

Cost Function (Mean Squared Error):

J(w, b) = (1 / 2m) * Σ (ŷ(i) - y(i))²

Gradient Descent Update Rules:

∂J/∂w = (1 / m) * Σ (ŷ(i) - y(i)) * x(i)
∂J/∂b = (1 / m) * Σ (ŷ(i) - y(i))

🧪 What I Did
Loaded the Fish dataset

Selected Length2 and Weight as feature and label

Applied gradient descent to learn w and b

Predicted weight values using the final learned parameters

Plotted:

Actual vs Predicted Weights

Regression line on scatter plot

📈 Final Learned Parameters
After training, I got the following values:


w = 16.2444
b = -5.3342
And predictions were made like this:


def predict(x, w, b):
    return w * x + b
    
📉 Results
The model successfully fit a line to the data.

The predicted weights followed the trend, but some predictions were off by a significant margin.

❗ Limitations
This is just a basic linear model using a single feature, so:

It may underfit the data (not capture the true curve)

It doesn't handle outliers or nonlinear patterns

It lacks validation techniques and feature scaling

🔧 Next Steps to Improve Accuracy
In future versions, I plan to:

Add polynomial features to capture non-linearity

Use multiple variables (like height, width) for multivariate regression

Apply feature scaling and regularization

Split into train/test and evaluate using R² and RMSE

📂 How to Run
Clone the repo

Open the notebook in Google Colab

Upload or fetch the CSV

Run the cells to train and predict

📌 Credits
Fish Market Dataset – Kaggle

Inspired by the "Andrew Ng ML" Specialization on Coursera
