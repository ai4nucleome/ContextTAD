#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import plot_exp1 as m


def main() -> None:
    if not m.CTCF_DATA.exists():
        raise FileNotFoundError(f"missing CTCF ChIA-PET file: {m.CTCF_DATA}")

    l0_support, l0_nosupport, l1_support, l1_nosupport = [], [], [], []
    for tool in m.TOOLS:
        folder, bed_base = m.TOOL_PATHS[tool]
        l0_file = m.DATA_DIR / folder / f"{bed_base}.L0"
        l1_file = m.DATA_DIR / folder / f"{bed_base}.L1+"

        supported, unsupported = m.count_supported_tads(m.CTCF_DATA, l0_file, m.RESOLUTION)
        l0_support.append(supported)
        l0_nosupport.append(unsupported)

        supported, unsupported = m.count_supported_tads(m.CTCF_DATA, l1_file, m.RESOLUTION)
        l1_support.append(supported)
        l1_nosupport.append(unsupported)

    data = pd.DataFrame(
        {
            "tool": m.TOOLS,
            "CTCF supported level 0 TAD": l0_support,
            "CTCF supported level 1+ TAD": l1_support,
            "unsupported level 0 TAD": l0_nosupport,
            "unsupported level 1+ TAD": l1_nosupport,
        }
    )
    m.OUTPUT_DIR.mkdir(exist_ok=True)
    out_csv = m.OUTPUT_DIR / "exp1_results.csv"
    data.to_csv(out_csv, index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
