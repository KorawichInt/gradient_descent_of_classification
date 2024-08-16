import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
import seaborn as sns

# Sigmoid function for logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (Log Loss for binary classification)
def log_loss(y, y_predict):
    return -(1/len(y)) * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))

# Update weights with gradient descent method for logistic regression
def gradient_descent(X, y, weights, lr=0.01, iterations=10000, stop_th=0.00000001):
    n = len(y)
    previous_cost = None
    cost_log = []
    X = np.c_[np.ones(n), X]  # Add bias term (intercept) to feature matrix
    
    for i in range(iterations):
        z = X.dot(weights)  # Matrix multiplication
        y_predict = sigmoid(z)

        # Calculate the cost (Log Loss)
        cost = log_loss(y, y_predict)

        # If the change in cost function is less than the threshold, break the loop
        if previous_cost and abs(previous_cost - cost) <= stop_th:
            break

        # Save the current cost function value
        previous_cost = cost

        # Gradient calculation
        gradient = (1 / n) * X.T.dot(y_predict - y)  # Compute gradient

        # Update weights
        weights -= lr * gradient

        cost_log.append(cost)
        if (i+1) % 1000 == 0:
            print(f"Iteration #{i+1} -> Cost = {cost}")

    print(f">>> Stopped training after {i+1} iterations\n")
    return weights, cost, y_predict, cost_log

if __name__ == "__main__":
    # path = "D:/datasets/dataset/heart_disease_health_indicators_BRFSS2015.csv"
    path = "heart_disease_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(path)

    # Randomly sample 10,000 rows
    # df = df.sample(n=100000, random_state=42)

    # Extract target variable and features
    y = np.array(df.iloc[:, 0])  # First column is the target
    X = df.iloc[:, 1:].values   # Remaining columns are features

    # Normalize features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize weights
    weights = np.zeros(X_train.shape[1] + 1)  # +1 for the bias term

    # Run gradient descent
    results = gradient_descent(X_train, y_train, weights)
    weights, cost, y_predict_prob, cost_log = results

    # Convert probabilities to binary labels (threshold = 0.5)
    y_predict_class = np.where(y_predict_prob >= 0.5, 1, 0)

    # Calculate accuracy on test data
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]  # Add bias term to test data
    y_test_predict_prob = sigmoid(X_test.dot(weights))
    y_test_predict_class = np.where(y_test_predict_prob >= 0.5, 1, 0)
    accuracy = accuracy_score(y_test, y_test_predict_class)
    print(f"Accuracy: {accuracy:.4f}")

    # Plot cost over iterations (Error graph)
    plt.figure(figsize=(10, 5))
    plt.plot(cost_log, label="Error (Log Loss)")
    plt.title("Error over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Error (Log Loss)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_predict_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_predict_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.show()