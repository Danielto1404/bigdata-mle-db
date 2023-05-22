import re
import typing as tp

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

import constants

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))


def remove_stopwords(text: str) -> str:
    """
    Removes stopwords from text
    :param text: str - text to remove stopwords from
    :return: str - text without stopwords
    """
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if w not in stop_words]
    return " ".join(filtered_text)


def remove_symbols(text: str) -> str:
    """
    Removes symbols from text
    :param text: str - text to remove symbols from
    :return: str - text without symbols
    """
    return re.sub(r"[^\w\s]", "", text)


def remove_numbers(text: str) -> str:
    """
    Removes numbers from text
    :param text: str - text to remove numbers from
    :return: str - text without numbers
    """
    return re.sub(r"\d+", "", text)


def remove_single_characters(text: str) -> str:
    """
    Removes single characters from text
    :param text: str - text to remove single characters from
    """
    return re.sub(r"\b[a-zA-Z]\b", "", text)


def remove_multiple_spaces(text: str) -> str:
    """
    Removes multiple spaces from text
    :param text: str - text to remove multiple spaces from
    :return: str - text without multiple spaces
    """
    return re.sub(r"\s+", " ", text)


def remove_html_tags(text: str) -> str:
    """
    Removes html tags from text
    :param text: str - text to remove html tags from
    :return: str - text without html tags
    """
    return re.sub(r"<.*?>", "", text)


def remove_urls(text: str) -> str:
    """
    Removes urls from text
    :param text: str - text to remove urls from
    :return: str - text without urls
    """
    return re.sub(r"http\S+", "", text)


def remove_mentions(text: str) -> str:
    """
    Removes mentions from text
    :param text: str - text to remove mentions from
    :return: str - text without mentions
    """
    return re.sub(r"@\S+", "", text)


def remove_hashtags(text: str) -> str:
    """
    Removes hashtags from text
    :param text: str - text to remove hashtags from
    :return: str - text without hashtags
    """
    return re.sub(r"#\S+", "", text)


def remove_emojis(text: str) -> str:
    """
    Removes emojis from text
    :param text: str - text to remove emojis from
    :return: str - text without emojis
    """
    return re.sub(r"\\x\S+", "", text)


def clean_text(text: str) -> str:
    """
    Cleans text
    :param text: str - text to clean
    :return: str - cleaned text
    """
    text = text.lower()
    text = remove_symbols(text)
    text = remove_numbers(text)
    text = remove_single_characters(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_emojis(text)
    text = remove_multiple_spaces(text)
    return text


def split_for_validation(train_df: pd.DataFrame, split_seed=42) -> tp.Tuple[
    pd.Series, pd.Series, pd.Series, pd.Series
]:
    """
    Splits train dataframe into train and validation dataframes
    :param train_df: DataFrame - train dataframe
    :param split_seed: int - seed for train_test_split
    :return: x_train, y_train, x_val, y_val - 4 dataframes for train and validation
    """
    x = train_df[constants.TEXT_COLUMN]
    y = train_df[constants.LABEL_COLUMN]
    return train_test_split(x, y, stratify=y, test_size=0.2, shuffle=True, random_state=split_seed)


__all__ = [
    "clean_text",
    "split_for_validation"
]
