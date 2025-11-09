import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv = RAW_DIR / "dataset.csv"
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        out = PROCESSED_DIR / "dataset.parquet"
        if out.exists():
            out.unlink()
        df.to_parquet(out)
        print(f"Processed saved: {out}")
    else:
        print(f"Missing: {raw_csv}")


if __name__ == "__main__":
    main()
