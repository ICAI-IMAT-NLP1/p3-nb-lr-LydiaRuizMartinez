from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    # TODO: Open the file, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object
    examples: List[SentimentExample] = []

    # open the file and read it line by line
    with open(infile, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # split the line into sentence and label using tab as delimiter
            parts: List[str] = line.split("\t")
            if len(parts) != 2:
                continue

            # extract sentence and label
            sentence: str = parts[0]
            label: int = int(parts[1])

            # tokenize the sentence
            tokenized_sentence: List[str] = tokenize(sentence)
            examples.append(SentimentExample(tokenized_sentence, label))

    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # TODO: Count unique words in all the examples from the training set
    vocab: Dict[str, int] = {}

    # extract all words from the SentimentExample objects and count occurrences of each word
    words: List[str] = [word for example in examples for word in example.words]
    word_counts: Counter[str] = Counter(words)

    # assign a unique index to each word in the vocabulary
    for word in word_counts:
        vocab[word] = len(vocab)

    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # TODO: Converts list of words into BoW, take into account the binary vs full
    bow: torch.Tensor = torch.zeros(len(vocab))

    # iterate over each word in the input text
    for word in text:
        if word in vocab:
            idx: int = vocab[word]

            # binary representation: set the index to 1 if the word appears at least once
            if binary:
                bow[idx] = 1

            # full BoW representation: Increment count for each occurrence of the word
            else:
                bow[idx] += 1

    return bow
