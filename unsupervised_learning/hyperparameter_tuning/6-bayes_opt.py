#!/usr/bin/env python3
"""Script to optimize a ML
model using GPyOpt"""


import GPyOpt
import numpy as np


# Define the function to optimize
def model_optimizer(x):
    # Define the hyperparameters to optimize
    learning_rate = x[:, 0]
    num_units = x[:, 1]
    dropout_rate = x[:, 2]
    l2_reg_weight = x[:, 3]
    batch_size = x[:, 4]

    # Train the model with the given hyperparameters
    # Return the value of the satisficing metric to optimize
    return -1 * model.train(learning_rate, num_units, dropout_rate, l2_reg_weight, batch_size)

# Define the bounds for the hyperparameters
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.001, 0.1)},
    {'name': 'num_units', 'type': 'discrete', 'domain': (16, 32, 64, 128)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'l2_reg_weight', 'type': 'continuous', 'domain': (0.001, 0.1)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128, 256)}
]

# Define the optimization objective
objective = GPyOpt.core.task.SingleObjective(model_optimizer)

# Define the initial design
initial_design = GPyOpt.experiment_design.initial_design('random', bounds, 10)

# Define the optimization algorithm
optimizer = GPyOpt.methods.BayesianOptimization(f=objective.f, domain=bounds, initial_design=initial_design, acquisition_type='EI', exact_feval=True)

# Run the optimization
max_iter = 30
optimizer.run_optimization(max_iter=max_iter, eps=1e-6)

# Save the best checkpoint
best_hyperparams = optimizer.X[np.argmin(optimizer.Y)]
model.save_checkpoint(best_hyperparams)

# Plot the convergence
optimizer.plot_convergence()

# Save the optimization report
with open('bayes_opt.txt', 'w') as f:
    f.write(optimizer.get_evaluations())
