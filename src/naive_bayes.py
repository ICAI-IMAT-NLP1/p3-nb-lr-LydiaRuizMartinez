import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # TODO: Estimate class priors and conditional probabilities of the bag of words
        self.class_priors: Dict[int, torch.Tensor] = self.estimate_class_priors(labels)
        self.conditional_probabilities: Dict[int, torch.Tensor] = (
            self.estimate_conditional_probabilities(features, labels, delta)
        )
        self.vocab_size: int = features.shape[1]
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # TODO: Count number of samples for each output class and divide by total of samples

        # convert the tensor of labels into a list and count occurrences of each class
        label_counts: Counter[int] = Counter(labels.tolist())

        # get the total number of samples
        total_samples: int = labels.shape[0]

        # compute prior probabilities for each class
        class_priors: Dict[int, torch.Tensor] = {
            label: torch.tensor(count / total_samples)
            for label, count in label_counts.items()
        }

        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # TODO: Estimate conditional probabilities for the words in features and apply smoothing

        # get unique classes and vocabulary size
        unique_classes = torch.unique(labels)
        vocab_size: int = features.shape[1]

        # initialize structures to store word counts per class and total words per class
        class_word_counts: Dict[int, torch.Tensor] = {
            c.item(): torch.zeros(vocab_size, dtype=torch.float32)
            for c in unique_classes
        }
        total_words_per_class: Dict[int, float] = {
            c.item(): 0.0 for c in unique_classes
        }

        # accumulate word counts per class
        for label, feature_vector in zip(labels.tolist(), features):
            class_word_counts[label] += feature_vector
            total_words_per_class[label] += feature_vector.sum().item()

        # compute conditional probabilities with Laplace smoothing
        class_conditional_probs: Dict[int, torch.Tensor] = {
            c: (class_word_counts[c] + delta)
            / (total_words_per_class[c] + delta * vocab_size)
            for c in unique_classes.tolist()
        }

        return class_conditional_probs

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # TODO: Calculate posterior based on priors and conditional probabilities of the words

        num_classes: int = len(self.class_priors)
        log_posteriors: torch.Tensor = torch.zeros(num_classes)

        # compute log posterior for each class
        for class_label, prior_prob in self.class_priors.items():
            # start with the log of the prior probability for the class
            log_posterior: torch.Tensor = torch.log(prior_prob)

            # compute the sum of log probabilities for each word in the feature vector
            conditional_probs: torch.Tensor = self.conditional_probabilities[
                class_label
            ]

            # apply log probabilities only where word counts are non-zero
            log_posterior += torch.sum(torch.log(conditional_probs) * feature)

            # store the computed log posterior probability
            log_posteriors[int(class_label)] = log_posterior

        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and obtain the class of maximum likelihood

        # compute log posterior probabilities for each class
        log_posteriors: torch.Tensor = self.estimate_class_posteriors(feature)

        # select the class with the highest log probability (maximum a posteriori decision rule)
        predicted_class: int = torch.argmax(log_posteriors).item()

        return predicted_class

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and transform them to probabilities (softmax)

        # compute log posterior probabilities
        log_posteriors: torch.Tensor = self.estimate_class_posteriors(feature)

        # convert log probabilities to normalized probabilities using softmax
        probs: torch.Tensor = torch.softmax(log_posteriors, dim=0)

        return probs
