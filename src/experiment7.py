
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

# Split dataset: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Create baseline SVM model
baseline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC())
])

# Train and predict
baseline_model.fit(X_train, y_train)
y_pred = baseline_model.predict(X_test)

# Evaluate model
print("\nBaseline SVM Performance")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
print("\n--- Grid Search ---")

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC())
])

param_grid = {
    "classifier__C": [0.1, 1, 10, 100],
    "classifier__kernel": ["linear", "rbf"],
    "classifier__gamma": ["scale", "auto"]
}

grid_search = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Grid Search Parameters:")
print(grid_search.best_params_)

print("Best Cross-Validation Score:")
print(grid_search.best_score_)

grid_pred = grid_search.predict(X_test)
print("Grid Search Test Accuracy:", accuracy_score(y_test, grid_pred))


# Random Search
print("\n--- Random Search ---")

param_dist = {
    "classifier__C": [0.01, 0.1, 1, 10, 100, 1000],
    "classifier__kernel": ["linear", "rbf"],
    "classifier__gamma": [
        "scale", "auto", 0.001, 0.01, 0.1, 1
    ]
}

random_search = RandomizedSearchCV(
    svm_pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best Random Search Parameters:")
print(random_search.best_params_)

print("Best Cross-Validation Score:")
print(random_search.best_score_)

random_pred = random_search.predict(X_test)
print("Random Search Test Accuracy:", accuracy_score(y_test, random_pred))

# MLflow Experiment Tracking
print("\n--- MLflow Experiment Tracking ---")

mlflow.set_experiment("Machine_Learning_Model_Comparison")

# Track Baseline SVM
with mlflow.start_run(run_name="Baseline_SVM"):
    baseline_model.fit(X_train, y_train)
    prediction = baseline_model.predict(X_test)

    mlflow.log_param("model", "SVM")
    mlflow.log_param("type", "Baseline")

    mlflow.log_metric("accuracy", accuracy_score(y_test, prediction))
    mlflow.log_metric("precision", precision_score(y_test, prediction))
    mlflow.log_metric("recall", recall_score(y_test, prediction))
    mlflow.log_metric("f1_score", f1_score(y_test, prediction))

    mlflow.sklearn.log_model(baseline_model, "baseline_model")

print("Baseline experiment recorded successfully.")

# Track Grid Search
with mlflow.start_run(run_name="Grid_Search_SVM"):
    prediction = grid_search.predict(X_test)

    mlflow.log_param("search_method", "Grid Search")

    for parameter, value in grid_search.best_params_.items():
        mlflow.log_param(parameter, value)

    mlflow.log_metric("accuracy", accuracy_score(y_test, prediction))
    mlflow.log_metric("precision", precision_score(y_test, prediction))
    mlflow.log_metric("recall", recall_score(y_test, prediction))
    mlflow.log_metric("f1_score", f1_score(y_test, prediction))

    mlflow.sklearn.log_model(
        grid_search.best_estimator_,
        "grid_search_model"
    )

print("Grid Search experiment recorded successfully.")

# Track Random Search
with mlflow.start_run(run_name="Random_Search_SVM"):
    prediction = random_search.predict(X_test)

    mlflow.log_param("search_method", "Random Search")

    for parameter, value in random_search.best_params_.items():
        mlflow.log_param(parameter, value)

    mlflow.log_metric("accuracy", accuracy_score(y_test, prediction))
    mlflow.log_metric("precision", precision_score(y_test, prediction))
    mlflow.log_metric("recall", recall_score(y_test, prediction))
    mlflow.log_metric("f1_score", f1_score(y_test, prediction))

    mlflow.sklearn.log_model(
        random_search.best_estimator_,
        "random_search_model"
    )

print("Random Search experiment recorded successfully.")

# Comparative Analysis
print("\n--- Model Comparison ---")

models = {
    "Baseline SVM": baseline_model,
    "Grid Search SVM": grid_search.best_estimator_,
    "Random Search SVM": random_search.best_estimator_
}

results = []

for name, model in models.items():
    prediction = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction),
        "Recall": recall_score(y_test, prediction),
        "F1 Score": f1_score(y_test, prediction)
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

# Plot comparison graph
results_df.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Comparison of Machine Learning Models")
plt.ylabel("Performance Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()


