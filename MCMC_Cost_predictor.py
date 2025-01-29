import numpy as np
import pymc as pm

class MarkovChain_MonteCarlo:
    def __init__(self, n):
        self.n = n
        self.model = None
        self.trace = None

    def fit(self, data):
        with pm.Model() as self.model:
            # Priors
            slope = pm.Normal('slope', mu=0, sigma=10)
            intercept = pm.Normal('intercept', mu=0, sigma=10)
            noise = pm.HalfNormal('noise', sigma=1)

            # Transition model
            def transition_model(t, previous_data):
                return slope * previous_data[t-1] + intercept + noise

            # Likelihood
            for t in range(self.n, len(data)):
                pm.Normal(f'obs_{t}', mu=transition_model(t, data), sigma=noise, observed=data[t])

            # Sampling
            self.trace = pm.sample(2000, tune=1000, cores=2)

    def predict(self, num_years):
        if self.trace is None:
            raise ValueError("Model has not been fitted yet. Please call fit() first.")

        predictions = np.zeros(num_years)
        for t in range(num_years):
            if t < self.n:
                # If we don't have enough previous data points, use the available ones
                predictions[t] = np.mean(self.trace['obs_' + str(t + self.n)])
            else:
                # Otherwise, use the previous n data points
                previous_data = predictions[t-self.n:t]
                predictions[t] = np.mean(self.trace['slope'] * previous_data + self.trace['intercept'] + self.trace['noise'])

        return predictions

# Example usage:
if __name__ == "__main__":
    # Example data
    data = np.array([20, 22, 23, 25, 27, 30, 32, 35])

    # Initialize and fit the model
    model = MarkovChain_MonteCarlo(n=3)
    model.fit(data)

    # Make predictions for the next 5 years
    num_years = 5
    predictions = model.predict(num_years)
    print("Predictions for the next 5 years:", predictions)
