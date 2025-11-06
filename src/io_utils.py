import os, io, requests
import pandas as pd

def fetch_csv_from_url(url: str, timeout: int = 60) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def load_public_hbn():
    url = os.getenv("HBN_PUBLIC_CSV_URL")
    if not url:
        raise RuntimeError("Set HBN_PUBLIC_CSV_URL env var before running.")
    return fetch_csv_from_url(url)

