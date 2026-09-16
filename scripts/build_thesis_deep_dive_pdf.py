from __future__ import annotations

import argparse
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "results" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "thesis" / "THESIS_DEEP_DIVE_RU.md"
DEFAULT_OUTPUT = REPO_ROOT / "thesis" / "THESIS_DEEP_DIVE_RU.pdf"

PAGE_WIDTH = 8.27
PAGE_HEIGHT = 11.69
LEFT_MARGIN = 0.08
RIGHT_MARGIN = 0.92
TOP_MARGIN = 0.94
BOTTOM_MARGIN = 0.06


@dataclass
class Block:
    kind: str
    text: str
    level: int = 0


STYLES = {
    "cover_title": {
        "fontsize": 22,
        "weight": "bold",
        "x": 0.08,
        "line_height": 0.05,
        "space_after": 0.03,
        "wrap": 38,
    },
    "cover_subtitle": {
        "fontsize": 12,
        "weight": "normal",
        "x": 0.08,
        "line_height": 0.028,
        "space_after": 0.02,
        "wrap": 80,
    },
    "h1": {
        "fontsize": 18,
        "weight": "bold",
        "x": 0.08,
        "line_height": 0.034,
        "space_after": 0.016,
        "wrap": 48,
    },
    "h2": {
        "fontsize": 15,
        "weight": "bold",
        "x": 0.08,
        "line_height": 0.030,
        "space_after": 0.014,
        "wrap": 60,
    },
    "h3": {
        "fontsize": 12.5,
        "weight": "bold",
        "x": 0.08,
        "line_height": 0.026,
        "space_after": 0.012,
        "wrap": 74,
    },
    "paragraph": {
        "fontsize": 10.5,
        "weight": "normal",
        "x": 0.08,
        "line_height": 0.021,
        "space_after": 0.010,
        "wrap": 98,
    },
    "bullet": {
        "fontsize": 10.5,
        "weight": "normal",
        "x": 0.11,
        "line_height": 0.021,
        "space_after": 0.007,
        "wrap": 92,
        "prefix": "• ",
    },
    "numbered": {
        "fontsize": 10.5,
        "weight": "normal",
        "x": 0.11,
        "line_height": 0.021,
        "space_after": 0.007,
        "wrap": 92,
    },
    "quote": {
        "fontsize": 10.5,
        "weight": "normal",
        "x": 0.10,
        "line_height": 0.021,
        "space_after": 0.010,
        "wrap": 90,
        "color": "#303030",
    },
    "code": {
        "fontsize": 9.5,
        "weight": "normal",
        "x": 0.10,
        "line_height": 0.019,
        "space_after": 0.010,
        "wrap": 90,
        "family": "DejaVu Sans Mono",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a readable PDF from THESIS_DEEP_DIVE_RU.md."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to the markdown source file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to the output PDF file.",
    )
    return parser.parse_args()


def clean_inline(text: str) -> str:
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", text)
    return " ".join(text.split())


def parse_markdown(markdown_text: str) -> list[Block]:
    blocks: list[Block] = []
    paragraph_lines: list[str] = []
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            text = clean_inline(" ".join(line.strip() for line in paragraph_lines))
            if text:
                blocks.append(Block(kind="paragraph", text=text))
            paragraph_lines = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                blocks.append(Block(kind="code", text="\n".join(code_lines)))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(raw_line)
            continue

        if not stripped:
            flush_paragraph()
            continue

        if stripped in {"---", "***", "___"}:
            flush_paragraph()
            blocks.append(Block(kind="rule", text=""))
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            blocks.append(
                Block(kind=f"h{level}", text=clean_inline(heading_match.group(2)), level=level)
            )
            continue

        quote_match = re.match(r"^>\s+(.*)$", stripped)
        if quote_match:
            flush_paragraph()
            blocks.append(Block(kind="quote", text=clean_inline(quote_match.group(1))))
            continue

        numbered_match = re.match(r"^(\d+\.)\s+(.*)$", stripped)
        if numbered_match:
            flush_paragraph()
            blocks.append(
                Block(
                    kind="numbered",
                    text=f"{numbered_match.group(1)} {clean_inline(numbered_match.group(2))}",
                )
            )
            continue

        bullet_match = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet_match:
            flush_paragraph()
            blocks.append(Block(kind="bullet", text=clean_inline(bullet_match.group(1))))
            continue

        paragraph_lines.append(raw_line)

    flush_paragraph()
    if code_lines:
        blocks.append(Block(kind="code", text="\n".join(code_lines)))

    return blocks


class PdfRenderer:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.pdf = PdfPages(self.output_path)
        self.page_number = 0
        self.fig = None
        self.y = TOP_MARGIN
        self._new_page()

    def _new_page(self) -> None:
        if self.fig is not None:
            self._save_page()
        self.page_number += 1
        self.fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        self.fig.patch.set_facecolor("white")
        self.y = TOP_MARGIN

    def _save_page(self) -> None:
        assert self.fig is not None
        self.fig.text(
            0.5,
            0.025,
            f"{self.page_number}",
            ha="center",
            va="bottom",
            fontsize=9,
            family="DejaVu Sans",
            color="#666666",
        )
        self.pdf.savefig(self.fig, bbox_inches="tight")
        plt.close(self.fig)

    def close(self) -> None:
        if self.fig is not None:
            self._save_page()
            self.fig = None
        self.pdf.close()

    def ensure_space(self, needed_height: float) -> None:
        if self.y - needed_height < BOTTOM_MARGIN:
            self._new_page()

    def add_cover(self, title: str) -> None:
        self.fig.text(
            STYLES["cover_title"]["x"],
            0.84,
            title,
            ha="left",
            va="top",
            fontsize=STYLES["cover_title"]["fontsize"],
            fontweight=STYLES["cover_title"]["weight"],
            family="DejaVu Sans",
        )
        subtitle_lines = [
            "Полный русскоязычный учебный конспект по дипломной работе",
            "о классификации респираторных звуков с помощью глубокого обучения.",
            "",
            "Формат документа:",
            "интуиция -> теория -> связь с твоим проектом -> практический смысл -> вопросы для защиты",
            "",
            f"Исходный файл: {DEFAULT_INPUT.name}",
            f"PDF-сборка: {self.output_path.name}",
        ]
        y = 0.72
        for line in subtitle_lines:
            self.fig.text(
                STYLES["cover_subtitle"]["x"],
                y,
                line,
                ha="left",
                va="top",
                fontsize=STYLES["cover_subtitle"]["fontsize"],
                family="DejaVu Sans",
                color="#303030",
            )
            y -= STYLES["cover_subtitle"]["line_height"]
        self._new_page()

    def add_rule(self) -> None:
        self.ensure_space(0.02)
        self.fig.lines.append(
            plt.Line2D(
                [LEFT_MARGIN, RIGHT_MARGIN],
                [self.y - 0.006, self.y - 0.006],
                transform=self.fig.transFigure,
                color="#B8B8B8",
                linewidth=0.8,
            )
        )
        self.y -= 0.018

    def add_block(self, block: Block) -> None:
        if block.kind == "rule":
            self.add_rule()
            return

        style = STYLES.get(block.kind, STYLES["paragraph"])
        family = style.get("family", "DejaVu Sans")
        color = style.get("color", "#111111")

        if block.kind == "code":
            lines = []
            for raw in block.text.splitlines():
                wrapped = textwrap.wrap(
                    raw if raw else " ",
                    width=style["wrap"],
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                )
                lines.extend(wrapped or [" "])
        else:
            text = block.text
            prefix = style.get("prefix", "")
            if block.kind == "bullet":
                wrapped = textwrap.wrap(
                    text,
                    width=style["wrap"],
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                lines = []
                for idx, line in enumerate(wrapped):
                    lines.append(f"{prefix}{line}" if idx == 0 else f"  {line}")
            else:
                lines = textwrap.wrap(
                    text,
                    width=style["wrap"],
                    break_long_words=True,
                    break_on_hyphens=False,
                ) or [""]

        needed_height = len(lines) * style["line_height"] + style["space_after"]
        self.ensure_space(needed_height)

        for line in lines:
            self.fig.text(
                style["x"],
                self.y,
                line,
                ha="left",
                va="top",
                fontsize=style["fontsize"],
                fontweight=style["weight"],
                family=family,
                color=color,
            )
            self.y -= style["line_height"]

        self.y -= style["space_after"]


def build_pdf(input_path: Path, output_path: Path) -> Path:
    markdown_text = input_path.read_text(encoding="utf-8")
    blocks = parse_markdown(markdown_text)

    if not blocks:
        raise ValueError(f"Source file is empty: {input_path}")

    renderer = PdfRenderer(output_path=output_path)
    title = blocks[0].text if blocks and blocks[0].kind == "h1" else "THESIS DEEP DIVE RU"
    renderer.add_cover(title)

    for index, block in enumerate(blocks):
        if index == 0 and block.kind == "h1":
            continue
        renderer.add_block(block)

    renderer.close()
    return output_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input markdown not found: {input_path}")

    pdf_path = build_pdf(input_path=input_path, output_path=output_path)
    print(f"PDF created: {pdf_path}")


if __name__ == "__main__":
    main()
