import os
from src.io_utils import fetch_csv_from_url

def test_fetch_public_hbn():
    url = os.getenv("HBN_PUBLIC_CSV_URL")
    assert url, "HBN_PUBLIC_CSV_URL not set"
    df = fetch_csv_from_url(url)
    assert not df.empty
    assert df.shape[0] >= 300
    print("OK:", df.shape)

if __name__ == "__main__":
    test_fetch_public_hbn()

