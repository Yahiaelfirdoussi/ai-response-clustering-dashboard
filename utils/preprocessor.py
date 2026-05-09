import re
import pandas as pd


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean"] = df["response"].apply(preprocess)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)
    return df


def extract_parquet_responses(path: str) -> pd.DataFrame:
    """Load a Chatbot Arena parquet and extract the first assistant turn from conversation_a."""
    raw_df = pd.read_parquet(path)

    def _first_assistant(conversation):
        for turn in conversation:
            if turn["role"] == "assistant":
                return turn["content"]
        return None

    raw_df["response"] = raw_df["conversation_a"].apply(_first_assistant)
    df = (
        raw_df[raw_df["language"] == "English"][["response"]]
        .dropna()
        .reset_index(drop=True)
    )
    return df
