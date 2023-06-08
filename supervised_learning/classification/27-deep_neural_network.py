#!/usr/bin/env python3
"""Deep Neural Network continued"""

import numpy as np
import pickle
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Defines deep neural network"""
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            self.__weights['W' + str(i+1)] = np.random.randn(
                layers[i], nx) * np.sqrt(2/nx)
            self.__weights['b' + str(i+1)] = np.zeros((layers[i], 1))
            nx = layers[i]

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Forward Propagation"""
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            Zi = np.dot(self.__weights['W' + str(i)],
                        self.__cache['A' + str(i - 1)]) +\
                        self.__weights['b' + str(i)]
            self.__cache['A' + str(i)] = self.sigmoid(Zi)
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Cost Function"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        for i in reversed(range(1, self.__L + 1)):
            A = cache['A' + str(i)]
            A_prev = cache['A' + str(i - 1)]
            W = self.__weights['W' + str(i)]
            if i == self.__L:
                dz = A - Y
            else:
                dz = da * self.sigmoid_derivative(A)
            dw = np.dot(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            if i > 1:
                da = np.dot(W.T, dz)
            self.__weights['W' + str(i)] -= alpha * dw
            self.__weights['b' + str(i)] -= alpha * db

    @staticmethod
    def sigmoid(Z):
        """activation"""
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def sigmoid_derivative(A):
        """derivative"""
        return A * (1 - A)

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the deep neural network by updating the weights and cache"""
        # Checking the types and values of iterations, alpha, and step
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be a positive and <= iterations")

        # Forward propagation, cost calculation, and back propagation in a loop
        cost_values = []
        for iteration in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            if verbose and iteration % step == 0:
                print("Cost after {} iterations: {}".format(iteration, cost))
            if graph and iteration % step == 0:
                cost_values.append(cost)

            self.gradient_descent(Y, cache, alpha)

        # Plot the training cost graph if graph=True
        if graph:
            import matplotlib.pyplot as plt
            plt.plot(range(0, iterations + 1, step), cost_values, 'b-')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None

    @staticmethod
    def sigmoid(Z):
        """Sigmoid Activation"""
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def sigmoid_derivative(A):
        """Sigmoid Derivative"""
        return A * (1 - A)
