import numpy as np

from src.utils import apply_sparsity, has_valid_eigenvalues, scale_matrix


class Reservoir:
    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        output_size: int,
        reservoir_values_range: float = 5.0,
        spectral_radius: float = 1.0,
        sparsity: float = 0.1,
        tmin: int = 10,
        leaky_rate: float = 0.8,
        feedback: bool = True,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        # Initialize the weight matrices
        self.Win = np.random.choice(
            [0, 1, -1], size=(reservoir_size, input_size), p=[0.5, 0.25, 0.25]
        )
        self.W = self._initialize_reservoir(
            reservoir_size, reservoir_values_range, spectral_radius, sparsity
        )
        self.Wout = np.zeros((output_size, reservoir_size))
        self.Wback = np.random.choice([-1, 1], size=(reservoir_size, output_size))

        # Initialize the internal state
        self.x = np.zeros(reservoir_size)
        self.tmin = tmin
        self.leaky_rate = leaky_rate
        self.feedback = feedback

    def _initialize_reservoir(self, size, range_value, spectral_radius, sparsity):
        valid_matrix = False
        while not valid_matrix:
            W = np.random.uniform(-range_value, range_value, (size, size))
            W = apply_sparsity(W, sparsity)
            if not has_valid_eigenvalues(W):
                continue
            W = scale_matrix(W, spectral_radius)

            if W is not None:
                valid_matrix = True
        return W

    def _activation(self, x):
        return np.tanh(x)

    def _output_activation(self, x):
        return x

    def update(self, u, y_prev):
        if self.feedback:
            new_x = self._activation(
                np.dot(self.Win, u)
                + np.dot(self.W, self.x)
                + np.dot(self.Wback, y_prev)
            )
        else:
            new_x = self._activation(np.dot(self.Win, u) + np.dot(self.W, self.x))
        self.x = (1 - self.leaky_rate) * self.x + self.leaky_rate * new_x

    def train(self, inputs, outputs, reg=0.01, noise_bound=0.01):
        states = []
        noise = np.random.uniform(-noise_bound, noise_bound, size=outputs.shape)
        outputs = outputs + noise  # Add noise to outputs
        # Update states with teacher forcing
        for t in range(self.tmin, len(inputs)):
            u = inputs[t]
            y_prev = outputs[t - 1] if t > 0 else np.zeros(outputs.shape[1])
            self.update(u, y_prev)
            states.append(self.x.copy())

        # Stack states into a T x N matrix (T: number of timesteps, N: reservoir size)
        states = np.vstack(states)

        # Solve for Wout using ridge regression: Wout = Y * X.T * (X * X.T + λI)^-1
        X = states.T
        Y = outputs[self.tmin :].T  # Discard transients from teacher outputs

        self.Wout = np.dot(
            Y, np.dot(X.T, np.linalg.pinv(np.dot(X, X.T) + reg * np.eye(X.shape[0])))
        )

        self.last_input = inputs[-1, :]

    def predict(self, n_steps=50):
        predictions = []
        y_pred = self.last_input
        for _ in range(n_steps):
            self.update(y_pred, y_pred)
            y_pred = self._output_activation(np.dot(self.Wout, self.x))
            predictions.append(y_pred)
        return np.array(predictions)
