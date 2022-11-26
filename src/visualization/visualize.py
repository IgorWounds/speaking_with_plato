from typing import List
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import nltk


def wordcloud(
    text: str, title: str, save_path: str, save: bool = True, collocations: bool = True
) -> None:
    """Creates a wordcloud graph of the text.

    Args:
        text (str): The text to be used to create the wordcloud.
        title (str): The title of the graph.
        save_path (str): The path to save the graph.
        save (bool, optional): Whether to save the graph. Defaults to True.
    """
    stopwords = set(STOPWORDS)
    stopwords.update("one")
    wordcloud = WordCloud(
        background_color="white",
        width=400,
        height=330,
        max_words=250,
        collocations=collocations,
        stopwords=stopwords,
    ).generate(text)
    # make image higher resolution
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(title)

    if save:
        plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.show()


def polarity_and_subjectivity(
    text: str, title: str, save_path: str, save: bool = True
) -> None:
    """Creates a graph of the polarity and subjectivity of the text."""

    blob = TextBlob(text)
    polarity = [sentence.sentiment.polarity for sentence in blob.sentences]
    subjectivity = [sentence.sentiment.subjectivity for sentence in blob.sentences]
    plt.plot(polarity, subjectivity, "ro")
    plt.title(title)
    plt.xlabel("Polarity")
    plt.ylabel("Subjectivity")
    if save:
        plt.savefig(save_path)
    plt.show()


def get_top_bigrams(
    text: str, title: str, n: int, stop_words: List[str], save_path: str
) -> None:
    """Creates a graph of the top n bigrams in the text."""
    text = " ".join([word for word in text.split() if word not in (stop_words)])
    bigrams = nltk.bigrams(text.split())
    bigram_freq = Counter(bigrams)
    bigram_freq = pd.DataFrame(bigram_freq.most_common(10), columns=["bigram", "freq"])
    bigram_freq["bigram"] = bigram_freq["bigram"].apply(lambda x: " ".join(x))
    plt.figure(figsize=(10, 6))
    # plot the top 10 bigrams as a vertical bar chart
    sns.barplot(x="freq", y="bigram", data=bigram_freq)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
