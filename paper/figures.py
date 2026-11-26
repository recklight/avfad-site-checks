"""pgfplots figures of the paper, written from the result files.

    fig_c1_gaps.tex      C1 gap, (i) minus (ii), for every detector and material, with 95% CIs
    fig_inside_hsa.tex   UAR on all of AVFAD against UAR inside HSA, per detector

Only group-level numbers go into the figures. Running it twice gives the same bytes.

    .venv/Scripts/python.exe -I -X utf8 -B paper/figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER.parent / "experiments" / "results"
LATEX = PAPER / "latex"
HEADER = "% Written by paper/figures.py from the result files; do not edit by hand.\n"


def load(relative: str) -> dict:
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def gap_rows() -> list[tuple[str, dict, bool]]:
    """(label, gap record, pre-registered) in the order the figure lists them."""
    hand = load("speech/c1_handcrafted.json")
    hubert = load("speech/c1_hubert.json")
    ribas = load("ribas_method/c1.json")
    ribas_balanced = load("ribas_method/c1_balanced.json")
    return [
        ("Handcrafted, /a/ (C1)", load("first_batch/summary.json")["c1"], True),
        ("HuBERT, /a/", load("sensitivity/hubert/c1_summary.json"), False),
        ("Handcrafted, sentences", hand["cape_v"], False),
        ("Handcrafted, reading", hand["reading"], False),
        ("HuBERT, sentences", hubert["cape_v"], False),
        ("HuBERT, reading", hubert["reading"], False),
        ("ComParE+SVM, /a/", ribas["a"]["gap_i_minus_ii"], False),
        ("ComParE+SVM, sentences", ribas["sentences"]["gap_i_minus_ii"], False),
        ("ComParE+SVM bal.$^{\\dagger}$, /a/", ribas_balanced["a"]["gap_i_minus_ii"], False),
        (
            "HuBERT fine-tuned$^{\\dagger}$, /a/",
            load("finetune/summary.json")["gap_i_minus_ii"],
            False,
        ),
    ]


def coordinate(y: int, record: dict) -> str:
    # Round the three ends first, so the bars end where the tables say.
    d, low, high = (round(100 * record[k], 1) for k in ("difference", "ci_low", "ci_high"))
    return f"({d:.1f},{y}) += ({high - d:.1f},0) -= ({d - low:.1f},0)"


def fig_c1_gaps() -> str:
    rows = gap_rows()
    labels = ",".join("{" + label + "}" for label, _, _ in rows)
    first = " ".join(coordinate(i + 1, r) for i, (_, r, pre) in enumerate(rows) if pre)
    rest = " ".join(coordinate(i + 1, r) for i, (_, r, pre) in enumerate(rows) if not pre)
    n = len(rows)
    return HEADER + (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        "  scale only axis, width=0.52\\columnwidth, height=4.8cm,\n"
        "  xmin=-2, xmax=40, ymin=0.4, ymax=" + f"{n + 0.6}" + ", y dir=reverse,\n"
        "  ytick={1,...," + str(n) + "}, yticklabels={" + labels + "},\n"
        "  xlabel={UAR gap, (i) minus (ii), points},\n"
        "  tick label style={font=\\footnotesize}, label style={font=\\footnotesize},\n"
        "  axis x line*=bottom, axis y line*=left, xmajorgrids, grid style={gray!25},\n"
        "]\n"
        "\\draw[gray] (axis cs:0,0.4) -- (axis cs:0," + f"{n + 0.6}" + ");\n"
        "\\draw[dashed, gray] (axis cs:10,0.4) -- (axis cs:10," + f"{n + 0.6}" + ");\n"
        "\\addplot[only marks, mark=*, mark size=1.8pt, error bars/.cd, x dir=both, x explicit]\n"
        "  coordinates {" + first + "};\n"
        "\\addplot[only marks, mark=o, mark size=1.8pt, error bars/.cd, x dir=both, x explicit]\n"
        "  coordinates {" + rest + "};\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}\n"
    )


def inside_rows() -> list[tuple[str, float, float]]:
    """(label, UAR on all of AVFAD, UAR inside HSA)."""
    every = load("exploratory/avfad_all.json")
    ribas = load("ribas_method/hsa.json")["a_n"]
    balanced = load("ribas_method/hsa_balanced.json")["a_n"]
    tuned = load("finetune/summary.json")
    return [
        (
            "Handcrafted",
            every["all_handcrafted"]["uar"],
            load("first_batch/summary.json")["c1"]["uar_ii"],
        ),
        (
            "HuBERT",
            every["all_hubert"]["uar"],
            load("sensitivity/hubert/ii_summary.json")["uar"],
        ),
        ("HuBERT fine-tuned$^{\\dagger}$", tuned["all"]["uar"], tuned["ii"]["uar"]),
        ("ComParE+SVM", ribas["everyone"]["uar"], ribas["ii"]["uar"]),
        ("ComParE+SVM bal.$^{\\dagger}$", balanced["everyone"]["uar"], balanced["ii"]["uar"]),
    ]


def fig_inside_hsa() -> str:
    rows = inside_rows()
    oracle = load("place_only/summary_oracle.json")["uar"]
    n = len(rows)
    labels = ",".join("{" + label + "}" for label, _, _ in rows)
    lines = "".join(
        f"\\draw[gray] (axis cs:{inside:.3f},{i + 1}) -- (axis cs:{everyone:.3f},{i + 1});\n"
        for i, (_, everyone, inside) in enumerate(rows)
    )
    everyone = " ".join(f"({e:.3f},{i + 1})" for i, (_, e, _) in enumerate(rows))
    inside = " ".join(f"({h:.3f},{i + 1})" for i, (_, _, h) in enumerate(rows))
    top, bottom = 0.4, n + 0.6
    return HEADER + (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        "  scale only axis, width=0.62\\columnwidth, height=3.4cm,\n"
        f"  xmin=0.45, xmax=0.95, ymin={top}, ymax={bottom}, y dir=reverse,\n"
        "  ytick={1,...," + str(n) + "}, yticklabels={" + labels + "},\n"
        "  xlabel={UAR},\n"
        "  tick label style={font=\\footnotesize}, label style={font=\\footnotesize},\n"
        "  axis x line*=bottom, axis y line*=left, xmajorgrids, grid style={gray!25},\n"
        "  legend style={font=\\footnotesize, at={(0.5,1.03)}, anchor=south, draw=none},\n"
        "  legend columns=2,\n"
        "]\n"
        f"\\draw[dotted, gray] (axis cs:0.5,{top}) -- (axis cs:0.5,{bottom});\n"
        f"\\draw[dashed, gray] (axis cs:{oracle:.3f},{top}) -- (axis cs:{oracle:.3f},{bottom});\n"
        + lines
        + "\\addplot[only marks, mark=*, mark size=1.8pt] coordinates {"
        + everyone
        + "};\n\\addlegendentry{All of AVFAD}\n"
        "\\addplot[only marks, mark=square, mark size=1.8pt] coordinates {"
        + inside
        + "};\n\\addlegendentry{Inside HSA}\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}\n"
    )


def main() -> None:
    LATEX.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("fig_c1_gaps.tex", fig_c1_gaps()),
        ("fig_inside_hsa.tex", fig_inside_hsa()),
    ):
        (LATEX / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote latex/{name}")


if __name__ == "__main__":
    main()
