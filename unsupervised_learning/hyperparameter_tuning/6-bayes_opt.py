#!/usr/bin/env python3
"""Script to optimize a ML model using GPyOpt"""

import GPyOpt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# Load the Iris dataset and split it into training and validation sets
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the function to optimize
def model_optimizer(x):
    n_estimators = int(x[:, 0])
    max_depth = int(x[:, 1]) if x[:, 1] != 0 else None
    min_samples_split = int(x[:, 2])
    min_samples_leaf = int(x[:, 3])
    max_features = x[:, 4]
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    return -accuracy

bounds = [
    {'name': 'n_estimators', 'type': 'discrete', 'domain': (50, 100, 150, 200)},
    {'name': 'max_depth', 'type': 'discrete', 'domain': (0, 10, 20, 30)},
    {'name': 'min_samples_split', 'type': 'discrete', 'domain': (2, 5, 10)},
    {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': (1, 2, 4)},
    {'name': 'max_features', 'type': 'continuous', 'domain': (0.1, 1.0)}
]

# Define the optimization algorithm
optimizer = GPyOpt.methods.BayesianOptimization(
    f=model_optimizer, 
    domain=bounds, 
    acquisition_type='EI'
)

max_iter = 30
eps = 1e-6
best_accuracy = np.inf

for i in range(max_iter):
    optimizer.run_optimization(max_iter=1)
    current_accuracy = np.min(optimizer.Y)
    
    if abs(current_accuracy - best_accuracy) < eps:
        print(f"Early stopping at iteration {i}, as improvement is less than {eps}.")
        break
    
    best_accuracy = current_accuracy
    best_hyperparams = optimizer.X[np.argmin(optimizer.Y)]
    
    model = RandomForestClassifier(
        n_estimators=int(best_hyperparams[0]),
        max_depth=int(best_hyperparams[1]) if best_hyperparams[1] != 0 else None,
        min_samples_split=int(best_hyperparams[2]),
        min_samples_leaf=int(best_hyperparams[3]),
        max_features=best_hyperparams[4],
        random_state=42,
    )
    
    model.fit(X_train, y_train)
    filename = f"checkpoint_n_estimators={best_hyperparams[0]}_max_depth={best_hyperparams[1]}_min_samples_split={best_hyperparams[2]}_min_samples_leaf={best_hyperparams[3]}_max_features={best_hyperparams[4]}.joblib"
    dump(model, filename)

# Plot the convergence
optimizer.plot_convergence()

# Save the optimization report
evaluations = optimizer.get_evaluations()
print(evaluations)

with open('bayes_opt.txt', 'w') as f:
    f.write(str(evaluations))
