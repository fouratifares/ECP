import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler


class Function:
    def __init__(self):
        self.dataset_name = "Yacht Hydrodynamics"
        self.data = self.load_dataset()
        self.X, self.y = self.preprocess_data()
        self.bounds = np.array([(-1, 1), (-1, 1)])  # ln(lambda), ln(sigma)
        self.dimensions = 2  # Two dimensions for ln(lambda) and ln(sigma)

    def load_dataset(self) -> pd.DataFrame:
        """Load the Yacht Hydrodynamics dataset from the UCI repository."""

        # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yacht-hydrodynamics/yacht_hydrodynamics.data"
        column_names = ['Longitudinal_Position', 'Prismatic_Coefficient', 'Length_Displacement_Ratio',
                        'Residuary_Resistance', 'Froude_Number', 'Wave_Resistance', 'Total_Resistance']
        data = pd.read_csv("./yacht_hydrodynamics.data", delim_whitespace=True, names=column_names, header=None, na_values='?')
        return data

    def preprocess_data(self) -> tuple:
        """Preprocess the data into features and target."""
        # Drop rows with NaN values
        data = self.data.dropna()

        # Extract features (X) and target (y)
        X = data.drop(columns=['Total_Resistance']).values  # Features (all columns except 'Total_Resistance')
        y = data['Total_Resistance'].values  # Target ('Total_Resistance')

        print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
        return X, y

    def __call__(self, x: np.ndarray = None, **kwargs) -> float:
        """Evaluate the model by performing kernel ridge regression."""

        if x is not None:
            # Handle input as a numpy array
            if len(x) != self.dimensions:
                raise ValueError(f"Input must have {self.dimensions} dimensions.")
        else:
            # Handle input as keyword arguments
            if len(kwargs) != self.dimensions:
                raise ValueError(f"Input must have {self.dimensions} dimensions.")
            x = np.array([kwargs[f'x{i}'] for i in range(self.dimensions)])

        x = np.array(x)  # Ensure x is a numpy array
        lambda_val = np.exp(x[0])
        sigma_val = np.exp(x[1])

        # Check if lambda_val and sigma_val are scalars or arrays
        if np.isscalar(lambda_val) and np.isscalar(sigma_val):
            lambda_vals = [lambda_val]
            sigma_vals = [sigma_val]
        else:
            lambda_vals = lambda_val
            sigma_vals = sigma_val

        all_rewards = []
        for lambda_, sigma_ in zip(lambda_vals, sigma_vals):
            kf = KFold(n_splits=3)
            mse_scores = []

            for train_index, test_index in kf.split(self.X):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                # Standardizing features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Model fitting
                model = KernelRidge(alpha=lambda_, kernel='rbf', gamma=1 / (2 * sigma_ ** 2))
                model.fit(X_train, y_train)

                # Predictions and MSE calculation
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_scores.append(mse)  # Collecting individual MSE for each fold

            mean_reward = -np.mean(mse_scores)

        return np.array(mean_reward)
