#!/usr/bin/env python3
import pandas as pd
import plot_exp16 as m


def main() -> None:
    for cell, path in m.CTCF_DATA.items():
        if not path.exists():
            raise FileNotFoundError(f"missing CTCF ChIA-PET file for {cell}: {path}")

    rows = []
    for cell in ["K562", "IMR90"]:
        target_file = m.CTCF_DATA[cell]
        for tool in m.TOOLS:
            folder, bed_base = m.TOOL_PATHS[cell][tool]
            l0_file = m.DATA_DIR / cell / folder / f"{bed_base}.L0"
            l1_file = m.DATA_DIR / cell / folder / f"{bed_base}.L1+"

            supported, unsupported = m.count_supported_tads(target_file, l0_file, m.RESOLUTION)
            l0_support = supported
            l0_nosupport = unsupported
            supported, unsupported = m.count_supported_tads(target_file, l1_file, m.RESOLUTION)
            l1_support = supported
            l1_nosupport = unsupported
            rows.append(
                {
                    "cell_line": cell,
                    "tool": tool,
                    "CTCF supported level 0 TAD": l0_support,
                    "CTCF supported level 1+ TAD": l1_support,
                    "unsupported level 0 TAD": l0_nosupport,
                    "unsupported level 1+ TAD": l1_nosupport,
                }
            )

    data = pd.DataFrame(rows)
    m.OUTPUT_DIR.mkdir(exist_ok=True)
    out_csv = m.OUTPUT_DIR / "exp7_results.csv"
    data.to_csv(out_csv, index=False)
    # backward-compatible filename
    data.to_csv(m.OUTPUT_DIR / "exp16_results.csv", index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
