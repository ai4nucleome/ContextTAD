#!/usr/bin/env python3
import pandas as pd
import plot_exp18 as m


def compute_cell_rows(cell_line: str):
    ctcf_left = m.load_ctcf_set(m.CTCF_FILES[cell_line]["left"])
    ctcf_right = m.load_ctcf_set(m.CTCF_FILES[cell_line]["right"])
    rows = []
    for tool in m.TOOLS:
        tads = m.load_tads(cell_line, tool)
        if tads.empty:
            left_hits = [0] * len(m.OFFSETS)
            right_hits = [0] * len(m.OFFSETS)
        else:
            left_hits = m.compute_hits(tads, ctcf_left, "start")
            right_hits = m.compute_hits(tads, ctcf_right, "end")
        for idx, offset in enumerate(m.OFFSETS):
            rows.append({"cell_line": cell_line, "side": "left", "tool": tool, "offset_kb": offset * 5, "count": int(left_hits[idx])})
            rows.append({"cell_line": cell_line, "side": "right", "tool": tool, "offset_kb": offset * 5, "count": int(right_hits[idx])})
    return rows


def main() -> None:
    all_rows = []
    for cell in ["K562", "IMR90"]:
        all_rows.extend(compute_cell_rows(cell))
    stats_df = pd.DataFrame(all_rows)
    m.OUTPUT_DIR.mkdir(exist_ok=True)
    out_csv = m.OUTPUT_DIR / "exp8_boundary_counts.csv"
    stats_df.to_csv(out_csv, index=False)
    # backward-compatible filename
    stats_df.to_csv(m.OUTPUT_DIR / "ctcf_chipseq_boundary_counts.csv", index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
