from LinearRegression import LinearRegression
import numpy as np

if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LinearRegression(X_train, y_train,learning_rate=0.01, n_iters=1000)
    regressor.fit()
    predictions = regressor.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print("sklearn MSE:", mse)
    
    print("mse", regressor.mse(y_test, predictions))
    
    print("sklearn MAE:", mae)
    
    print("mae", regressor.mae(y_test, predictions))
