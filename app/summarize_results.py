"""
Aggregate multiseed_results.json into a mean+/-std comparison table across all
configurations (single-task, joint, and cross-task variants) and print the key
transfer deltas. Writes models/comparison_summary.txt.
"""
import json
import statistics as st
from pathlib import Path

RES = Path("models/multiseed_results.json")
OUT = Path("models/comparison_summary.txt")

ORDER = ["seg_only", "pose_only", "joint", "joint_uw", "joint_mg", "joint_pg", "joint_bi"]
LABEL = {
    "seg_only":  "Seg-only (single)",
    "pose_only": "Pose-only (single)",
    "joint":     "Joint (baseline MTL)",
    "joint_uw":  "Joint + uncertainty wt.",
    "joint_mg":  "Joint + mask-guided (B1)",
    "joint_pg":  "Joint + pose-guided (B2)",
    "joint_bi":  "Joint + bidirectional",
}


def agg(values):
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0, len(xs))


def fmt(a):
    return "-" if a is None else f"{a[0]:.4f} +/- {a[1]:.4f} (n={a[2]})"


def main():
    if not RES.exists():
        print("no results file yet")
        return
    d = json.loads(RES.read_text())

    lines = ["=" * 78,
             f"{'Configuration':<26}{'mIoU':<24}{'PCK@0.05':<24}",
             "-" * 78]
    means = {}
    for c in ORDER:
        if c not in d:
            continue
        mi = agg([r.get("miou") for r in d[c].values()])
        pk = agg([r.get("pck") for r in d[c].values()])
        means[c] = (mi[0] if mi else None, pk[0] if pk else None)
        lines.append(f"{LABEL[c]:<26}{fmt(mi):<24}{fmt(pk):<24}")
    lines.append("=" * 78)

    so = means.get("seg_only", (None, None))[0]   # seg-only mIoU
    po = means.get("pose_only", (None, None))[1]   # pose-only PCK
    lines.append("\nTransfer analysis (vs single-task means):")
    for c in ["joint", "joint_uw", "joint_mg", "joint_pg", "joint_bi"]:
        if c not in means:
            continue
        jm, jp = means[c]
        seg_d = f"mIoU {jm:.4f} (vs seg-only {so:.4f}, {jm-so:+.4f})" if (jm is not None and so is not None) else ""
        pose_d = f"PCK {jp:.4f} (vs pose-only {po:.4f}, {jp-po:+.4f})" if (jp is not None and po is not None) else ""
        lines.append(f"  {LABEL[c]:<26} {seg_d}  |  {pose_d}")
    lines.append("\nPositive transfer = delta > 0 on BOTH; recovery = PCK delta closer to 0 than baseline joint.")

    out = "\n".join(lines)
    print(out)
    OUT.write_text(out)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
