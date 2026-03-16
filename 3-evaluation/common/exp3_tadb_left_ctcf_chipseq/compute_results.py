#!/usr/bin/env python3
import pandas as pd
import plot_exp7 as m


def main() -> None:
    if not m.DATA_DIR.exists():
        raise FileNotFoundError(f"missing data dir: {m.DATA_DIR}")
    if not m.CTCF_FILE.exists():
        raise FileNotFoundError(f"missing CTCF file: {m.CTCF_FILE}")

    stats_list = []
    for tool in m.TOOLS:
        match_num, total_num, fraction = m.calculate_ctcf_fraction(tool, m.DATA_DIR, m.CTCF_FILE, boundary="left")
        stats_list.append(
            {
                "tool": tool,
                "total_boundaries": total_num,
                "ctcf_supported": match_num,
                "ctcf_unsupported": total_num - match_num,
                "fraction": fraction,
            }
        )

    stats_df = pd.DataFrame(stats_list)
    m.OUTPUT_DIR.mkdir(exist_ok=True)
    out_csv = m.OUTPUT_DIR / "exp3_results.csv"
    stats_df.to_csv(out_csv, index=False)
    # backward-compatible filename
    stats_df.to_csv(m.OUTPUT_DIR / "exp7_results.csv", index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
