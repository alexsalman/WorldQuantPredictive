from abc import ABC, abstractmethod
import numpy as np

class BaseLinearModel(ABC):

    def __init__(self):
        self._theta = None

    @abstractmethod
    def fit(self, X: np.array, y: np.array) -> None:
        return

    @abstractmethod
    def predict(self, X: np.array) -> np.array:
        return

    def get_params(self):
        return self._theta[1:] if self._theta is not None else None


# Your code should begin here, do not alter the above class
# definition or imports.

# Alex Salman
# aalsalma@ucsc.edu
# Candidate for Machine Learning Engineer at WorldQuant Predictive
# 2/15/2022

# Please run this model from your terminal using
# the correct path to the file using the command
# python3 linear_model.py

class LinearLeastSquares(BaseLinearModel):

    def __init__(self):
        self.a = 0
        self.b = 0
        super().__init__()

    # fit method with learning_rate of 0.01 and 1000 iterations
    def fit(self, X, y, n, learning_rate=0.01, iterations=1000):
        # datapoints
        dp_no = float(len(y))
        # costs reduction values during iteration
        costs = []
        # y = a + bx with starting coordinates of a = 0, b = 0
        a = [0] * n
        b = 0
        y_pred = np.dot(a, np.transpose(X)) + b
        cost = np.sum((y - y_pred) ** 2) / dp_no
        print ('Starting cost : ' + str(cost))

        # iteration of the batches
        for i in range(iterations):

            # first derivative of the equation
            derivative_a = 0
            derivative_b = 0
            for j in range(len(y)):
                derivative_b += -(2 / dp_no) * (y[j] - ((np.dot(a, np.transpose(X[j]))) + b))
                derivative_a += -(2 / dp_no) * X[j] * (y[j] - ((np.dot(a, np.transpose(X[j]))) + b))

            # change a, b
            a = a - (learning_rate * derivative_a)
            b = b - (learning_rate * derivative_b)
            y_reg = np.dot(a, np.transpose(X)) + b
            y_reg = y_reg.ravel()

            new_cost = np.sum((y - y_reg) ** 2) / dp_no
            costs.append(new_cost)

            # find the convergence point of gradient descent
            if np.sum(costs[-3:])/costs[-1:] == 3:
                print ('Convergence point is found at iteration : ' + str(i))
                break

        print ("Coefficient : " + str(a))
        print ("Intercept : " + str(b))
        self.a = a
        self.b = b

    # predict method
    def predict(self, X):
        ans = np.dot(self.a, np.transpose(X)) + self.b
        return ans

def main():
    from sklearn.datasets import make_regression
    # number of features
    nn = 5
    X, y_true = make_regression(n_samples=100, n_features=nn, n_informative=1, noise=10, random_state=0)
    # create an object of subclass Linear Least Squares
    obj = LinearLeastSquares()
    # fit the model
    obj.fit(X, y_true, nn)
    # prediction
    y_pred = obj.predict(X)
    print(y_pred)
    # accuracy check
    MSE = np.square(np.subtract(y_true, y_pred)).mean()
    print('MSE: ', MSE)

if __name__ == "__main__":
    main()
