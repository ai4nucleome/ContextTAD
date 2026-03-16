#!/usr/bin/env python3
from pathlib import Path
import plot_exp6 as m


def main() -> None:
    df = m.load_data(m.DATA_FILE)
    processed_df = m.process_data(df)

    results_df = processed_df.copy().rename(columns={"TadsFile": "tool"})
    m.OUTPUT_DIR.mkdir(exist_ok=True)
    out_csv = m.OUTPUT_DIR / "exp2_results.csv"
    results_df.to_csv(out_csv, index=False)
    # backward-compatible filename
    results_df.to_csv(m.OUTPUT_DIR / "exp6_results.csv", index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
