import torch

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words
import torch


class LogisticRegression:
    def __init__(self, random_state: int):
        self._weights: torch.Tensor = None
        self.random_state: int = random_state

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float,
        epochs: int,
    ):
        """
        Train the logistic regression model using pre-processed features and labels.

        Args:
            features (torch.Tensor): The bag of words representations of the training examples.
            labels (torch.Tensor): The target labels.
            learning_rate (float): The learning rate for gradient descent.
            epochs (int): The number of iterations over the training dataset.

        Returns:
            None: The function updates the model weights in place.
        """
        # TODO: Implement gradient-descent algorithm to optimize logistic regression weights

        # initialize weights using the initialize_parameters method.
        self.weights: torch.Tensor = self.initialize_parameters(
            features.shape[1], self.random_state
        )

        # add a bias term to the feature matrix
        features_with_bias: torch.Tensor = torch.cat(
            (features, torch.ones((features.shape[0], 1))), dim=1
        )

        # training loop
        for epoch in range(epochs):
            # compute predictions using the sigmoid function
            logits: torch.Tensor = torch.matmul(features_with_bias, self.weights)
            predictions: torch.Tensor = self.sigmoid(logits)

            # compute loss using binary cross-entropy
            loss: torch.Tensor = self.binary_cross_entropy_loss(predictions, labels)

            # compute gradient of the loss with respect to the weights
            error: torch.Tensor = predictions - labels
            gradient: torch.Tensor = (
                torch.matmul(features_with_bias.T, error) / features_with_bias.shape[0]
            )

            # update weights
            self.weights -= learning_rate * gradient

        return

    def predict(self, features: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
        """
        Predict class labels for given examples based on a cutoff threshold.

        Args:
            features (torch.Tensor): The bag of words representations of the input examples.
            cutoff (float): The threshold for classifying a sample as positive. Defaults to 0.5.

        Returns:
            torch.Tensor: Predicted class labels (0 or 1).
        """
        # initialize a tensor to store the predicted class labels
        decisions: torch.Tensor = torch.zeros(features.shape[0])

        # compute the predicted probabilities for each sample
        probabilities: torch.Tensor = self.predict_proba(features)

        # apply the cutoff threshold to classify samples as 0 or 1
        decisions = (probabilities >= cutoff).float()

        return decisions

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predicts the probability of each sample belonging to the positive class using pre-processed features.

        Args:
            features (torch.Tensor): The bag of words representations of the input examples.

        Returns:
            torch.Tensor: A tensor of probabilities for each input sample being in the positive class.

        Raises:
            ValueError: If the model weights are not initialized (model not trained).
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call the 'train' method first.")

        # add bias term to feature matrix
        features_with_bias: torch.Tensor = torch.cat(
            (features, torch.ones((features.shape[0], 1))), dim=1
        )

        # compute probabilities using the sigmoid function
        probabilities: torch.Tensor = self.sigmoid(
            torch.matmul(features_with_bias, self.weights)
        )

        return probabilities

    def initialize_parameters(self, dim: int, random_state: int) -> torch.Tensor:
        """
        Initialize the weights for logistic regression using a normal distribution.

        This function initializes the weights (and bias as the last element) with values drawn from a normal distribution.
        The use of random weights can help in breaking the symmetry and improve the convergence during training.

        Args:
            dim (int): The number of features (dimension) in the input data.
            random_state (int): A seed value for reproducibility of results.

        Returns:
            torch.Tensor: Initialized weights as a tensor with size (dim + 1,).
        """
        # set random seed for reproducibility
        torch.manual_seed(random_state)

        # initialize weights using a normal distribution
        params: torch.Tensor = torch.randn(dim + 1)

        return params

    @staticmethod
    def sigmoid(z: torch.Tensor) -> torch.Tensor:
        """
        Compute the sigmoid of z.

        This function applies the sigmoid function, which is defined as 1 / (1 + exp(-z)).
        It is used to map predictions to probabilities in logistic regression.

        Args:
            z (torch.Tensor): A tensor containing the linear combination of weights and features.

        Returns:
            torch.Tensor: The sigmoid of z.
        """
        result: torch.Tensor = torch.sigmoid(z)
        return result

    @staticmethod
    def binary_cross_entropy_loss(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the binary cross-entropy loss.

        The binary cross-entropy loss is a common loss function for binary classification. It calculates the difference
        between the predicted probabilities and the actual labels.

        Args:
            predictions (torch.Tensor): Predicted probabilities from the logistic regression model.
            targets (torch.Tensor): Actual labels (0 or 1).

        Returns:
            torch.Tensor: The computed binary cross-entropy loss.
        """
        predictions = torch.clamp(predictions, min=1e-9, max=1 - 1e-9)
        ce_loss = -torch.mean(
            targets * torch.log(predictions)
            + (1 - targets) * torch.log(1 - predictions)
        )
        return ce_loss

    @property
    def weights(self):
        """Get the weights of the logistic regression model."""
        return self._weights

    @weights.setter
    def weights(self, value):
        """Set the weights of the logistic regression model."""
        self._weights: torch.Tensor = value
