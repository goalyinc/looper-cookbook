"""The tutorial notebooks, read and run without a Jupyter kernel.

A notebook is JSON: cells of Markdown or code, a code cell's tags and its
saved outputs. ``read`` gives the cells and ``run`` executes the code cells
in order in one namespace, as a kernel does top to bottom: top-level
``await`` runs on the caller's event loop, and IPython's magics, such as
``%pip``, are skipped. After the cell tagged ``parameters`` it replaces the
values that cell assigned, as papermill does, which is how a test points a
notebook at a server.

``record`` runs a notebook the same way and saves what each code cell
printed as its output, which is how a tutorial's outputs come from a hosted
deployment. Run as a module, it records one notebook::

    python -m looper_cookbook.tests.notebooks record PATH --base-url URL --project NAME
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import dataclasses
import inspect
import io
import json
import pathlib
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any, TextIO

__all__ = [
    "TUTORIALS",
    "Cell",
    "parameters",
    "prose",
    "python",
    "read",
    "record",
    "run",
    "tutorials",
]

TUTORIALS = pathlib.Path(__file__).parents[2] / "tutorials"


@dataclasses.dataclass(frozen=True, slots=True)
class Cell:
    """One cell of a notebook.

    Attributes:
        kind: ``markdown`` or ``code``.
        outputs: A code cell's saved outputs, as the notebook format holds them.
        attachments: Whether files are embedded in the cell.
    """

    kind: str
    source: str
    tags: tuple[str, ...]
    outputs: tuple[Mapping[str, Any], ...]
    attachments: bool


def tutorials() -> list[pathlib.Path]:
    """Every tutorial notebook, in page order."""
    return sorted(TUTORIALS.glob("*.ipynb"))


def read(path: pathlib.Path) -> list[Cell]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook["nbformat"] != 4:
        raise ValueError(f"{path.name} is nbformat {notebook['nbformat']}, not 4")
    return [
        Cell(
            kind=cell["cell_type"],
            source="".join(cell["source"]),
            tags=tuple(cell["metadata"].get("tags", ())),
            outputs=tuple(cell.get("outputs", ())),
            attachments=bool(cell.get("attachments")),
        )
        for cell in notebook["cells"]
    ]


def python(source: str) -> str:
    """A code cell's Python: its lines without IPython's magics and shell escapes."""
    return "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines()
    )


def prose(source: str) -> str:
    """A Markdown cell's text outside code: fenced blocks and code spans removed."""
    text = re.sub(r"^```.*?^```[^\n]*$", "", source, flags=re.DOTALL | re.MULTILINE)
    return re.sub(r"`[^`\n]*`", "", text)


def parameters(cells: Sequence[Cell]) -> dict[str, Any]:
    """The names the notebook's ``parameters`` cell assigns, with their values."""
    tagged = [cell for cell in cells if cell.kind == "code" and "parameters" in cell.tags]
    if len(tagged) != 1:
        raise ValueError(f"a tutorial has one cell tagged parameters, not {len(tagged)}")
    values = {}
    for statement in ast.parse(tagged[0].source).body:
        match statement:
            case ast.Assign(targets=[ast.Name(id=name)], value=value):
                # literal_eval raises on anything computed, which a replacement could not follow.
                values[name] = ast.literal_eval(value)
            case _:
                raise ValueError(
                    "a parameters cell holds plain assignments, not "
                    f"{ast.unparse(statement).splitlines()[0]!r}"
                )
    return values


class _Tee(io.StringIO):
    """Keeps what a cell prints, and passes it on as it comes, so a long run
    shows its progress and a failing one what it printed."""

    def __init__(self, terminal: TextIO) -> None:
        super().__init__()
        self._terminal = terminal

    def write(self, s: str, /) -> int:
        self._terminal.write(s)
        return super().write(s)

    def flush(self) -> None:
        self._terminal.flush()


async def run(path: pathlib.Path, replaced: Mapping[str, Any]) -> list[str]:
    """Runs the notebook's code cells in order and returns what each printed.

    Args:
        path: The notebook.
        replaced: Values for names its ``parameters`` cell assigns, set once
            that cell has run.

    Returns:
        Each code cell's standard output, in order. It reaches the caller's
        standard output too, as the cell prints it.
    """
    cells = read(path)
    unknown = set(replaced) - set(parameters(cells))
    if unknown:
        raise ValueError(f"{path.name}'s parameters cell assigns no {sorted(unknown)}")
    namespace: dict[str, Any] = {"__name__": "__main__"}
    code_cells = [cell for cell in cells if cell.kind == "code"]
    printed = []
    for number, cell in enumerate(code_cells, start=1):
        code = compile(
            python(cell.source),
            f"{path.name}, code cell {number}",
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        with contextlib.redirect_stdout(_Tee(sys.stdout)) as output:
            result = eval(code, namespace)
            if inspect.iscoroutine(result):
                await result
        printed.append(output.getvalue())
        if "parameters" in cell.tags:
            namespace.update(replaced)
    return printed


async def record(path: pathlib.Path, replaced: Mapping[str, Any]) -> None:
    """Runs the notebook as ``run`` does and saves what each code cell printed.

    Each code cell's output becomes its standard output, as one stream, and
    its execution count its place among the code cells, as Jupyter saves a
    run in a fresh kernel. The notebook is written only once every cell has
    run, and nothing else in it changes.

    Args:
        path: The notebook, rewritten in place.
        replaced: As ``run`` takes them. The saved cells keep their own values.
    """
    printed = await run(path, replaced)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for count, (cell, text) in enumerate(zip(code_cells, printed, strict=True), start=1):
        cell["execution_count"] = count
        stream = {"name": "stdout", "output_type": "stream", "text": text.splitlines(keepends=True)}
        cell["outputs"] = [stream] if text else []
    # Jupyter's own layout, so a recording's diff is its outputs alone.
    text = json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m looper_cookbook.tests.notebooks",
        description="Save a tutorial's outputs from a run against a hosted deployment.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    record_command = commands.add_parser(
        "record",
        help="run a notebook's code cells in order and save what each printed",
        description=(
            "Runs the notebook's code cells in order, with the key in LOOPER_API_KEY, and saves "
            "what each printed as its output. Leave out --base-url and --project for a notebook "
            "that calls no service."
        ),
    )
    record_command.add_argument("path", type=pathlib.Path, help="the notebook, rewritten in place")
    record_command.add_argument("--base-url", help="the API root, in place of its BASE_URL")
    record_command.add_argument("--project", help="the project, in place of its PROJECT")
    arguments = parser.parse_args()
    given = {"BASE_URL": arguments.base_url, "PROJECT": arguments.project}
    asyncio.run(record(arguments.path, {k: v for k, v in given.items() if v is not None}))


if __name__ == "__main__":
    main()
