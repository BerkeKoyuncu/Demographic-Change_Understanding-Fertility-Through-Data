import pandas as pd
from src.config import RAW_DIR, PROCESSED_DIR

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv = RAW_DIR / "dataset.csv"
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        out = PROCESSED_DIR / "dataset.parquet"
        if out.exists():
            out.unlink()
        df.to_parquet(out)
        print("Processed saved:", out)
    else:
        print(f"Missing: {raw_csv}")

if __name__ == "__main__":
    main()
