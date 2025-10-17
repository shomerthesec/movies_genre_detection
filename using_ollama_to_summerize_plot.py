from pathlib import Path
import logging
import re
from typing import List, Optional

import pandas as pd
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def summarize_plot(plot_text: str, url: str = "http://localhost:11434/api/generate", model: str = "qwen2.5:3b", timeout: int = 10) -> str:
    """Call a local generation API to produce a single non-spoiler sentence.

    This implementation intentionally contains only a single explicit prompt
    that instructs the model to avoid spoilers. No post-processing, heuristics,
    or retries are performed — the function simply returns the first text it
    can extract from the JSON response.
    """
    if not isinstance(plot_text, str) or not plot_text.strip():
        return ""

    prompt = (
        "You are a careful movie summarizer. Write exactly one concise sentence "
        "that captures the film's central premise, tone or theme WITHOUT revealing "
        "any spoilers, plot twists, endings, or character outcomes. Avoid phrases "
        "that disclose who dies, who is revealed to be a villain, or any event that "
        "would spoil the viewing experience. Use neutral, non-specific language.\n\n"
        "Good examples:\n"
        "- 'A devoted teacher fights to save her students while confronting her own past.'\n"
        "- 'An outsider uncovers corruption in a small town and must choose between safety and justice.'\n\n"
        "Bad examples (these are spoilers and should never be produced):\n"
        "- 'The hero dies at the end.'\n"
        "- 'It is revealed that the mayor was the killer.'\n\n"
        f"Plot:\n{plot_text}\n\n"
        "Now write the single non-spoiler sentence."
    )

    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload)
        return response.json().get("response", "").strip()
    except:
        return plot_text
    

def clean_text(text) -> str:
    """Normalize text for downstream processing.

    - lowercases
    - strips punctuation
    - collapses whitespace
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_data(df: pd.DataFrame, top_genres: List[str]) -> pd.DataFrame:
    """Filter and clean a movie DataFrame.

    - Collapse genres not in `top_genres` to 'other'.
    - Keep only genres with >50 examples.
    - Drop rows missing Title / Release Year / Plot and drop duplicates.
    """
    df = df.copy()
    df["Genre"] = df["Genre"].apply(lambda x: x if x in top_genres else "other")

    genre_counts = df["Genre"].value_counts()
    valid_genres = genre_counts[genre_counts > 50].index
    df = df[df["Genre"].isin(valid_genres)]

    df = df.dropna(subset=["Title", "Release Year", "Plot"]) 
    df = df.drop_duplicates(subset=["Title", "Release Year", "Plot"]) 
    return df

df = pd.read_csv("/media/shomer/Windows/Users/shomer/data/New folder/genre_project/data/wiki_movie_plots_deduped.csv")
top_genres = df['Genre'].value_counts().index[:25]
df = prepare_data(df, top_genres)
df['summerized_plot'] = df['Plot'].apply(summarize_plot)
df.to_csv("/media/shomer/Windows/Users/shomer/data/New folder/genre_project/tiny_bert_training/summerized_plots_without_spoliers.csv")
    