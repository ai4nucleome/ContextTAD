#!/usr/bin/env python3
import pandas as pd
import plot_exp14 as m


def main() -> None:
    if not m.CTCF_DATA.exists():
        raise FileNotFoundError(f"missing CTCF ChIA-PET file: {m.CTCF_DATA}")

    all_frames = []
    m.OUTPUT_DIR.mkdir(exist_ok=True)

    for cov_dir, cov_name in m.COVS.items():
        l0_support, l0_nosupport, l1_support, l1_nosupport = [], [], [], []
        for tool in m.TOOLS:
            folder, bed_tpl = m.TOOL_PATHS[tool]
            bed_base = bed_tpl.format(cov=cov_name)
            tool_dir = m.DATA_ROOT / cov_dir / folder
            l0_file = tool_dir / f"{bed_base}.L0"
            l1_file = tool_dir / f"{bed_base}.L1+"

            supported, unsupported = m.count_supported_tads(m.CTCF_DATA, l0_file, m.RESOLUTION)
            l0_support.append(supported)
            l0_nosupport.append(unsupported)

            supported, unsupported = m.count_supported_tads(m.CTCF_DATA, l1_file, m.RESOLUTION)
            l1_support.append(supported)
            l1_nosupport.append(unsupported)

        data = pd.DataFrame(
            {
                "cov": cov_name,
                "tool": m.TOOLS,
                "CTCF supported level 0 TAD": l0_support,
                "CTCF supported level 1+ TAD": l1_support,
                "unsupported level 0 TAD": l0_nosupport,
                "unsupported level 1+ TAD": l1_nosupport,
            }
        )
        all_frames.append(data)
        cov_csv = m.OUTPUT_DIR / f"exp6_results_{cov_name}.csv"
        data.to_csv(cov_csv, index=False)
        # backward-compatible filename
        data.to_csv(m.OUTPUT_DIR / f"exp14_results_{cov_name}.csv", index=False)

    if all_frames:
        summary = pd.concat(all_frames, axis=0, ignore_index=True)
        summary_file = m.OUTPUT_DIR / "exp6_results_all.csv"
        summary.to_csv(summary_file, index=False)
        # backward-compatible filename
        summary.to_csv(m.OUTPUT_DIR / "exp14_results_all.csv", index=False)
        print(summary_file)


if __name__ == "__main__":
    main()
