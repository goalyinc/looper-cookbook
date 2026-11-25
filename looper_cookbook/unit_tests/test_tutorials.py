"""What a tutorial notebook keeps to, so the docs site can render it from the
file alone."""

from __future__ import annotations

import copy
import json
import pathlib
import re
from typing import Any

import pytest

from looper_cookbook.tests import notebooks
from looper_sdk import __version__

TUTORIALS = notebooks.tutorials()
every_tutorial = pytest.mark.parametrize("path", TUTORIALS, ids=lambda path: path.stem)


def test_the_tutorials_are_found() -> None:
    assert TUTORIALS


@every_tutorial
def test_a_tutorial_is_named_by_its_number(path: pathlib.Path) -> None:
    assert re.fullmatch(r"[1-9]\d\d_[a-z0-9_]+\.ipynb", path.name)


@every_tutorial
def test_a_tutorial_opens_with_its_only_title(path: pathlib.Path) -> None:
    cells = notebooks.read(path)
    assert cells[0].kind == "markdown" and cells[0].source.startswith("# ")
    markdown = "\n".join(notebooks.prose(cell.source) for cell in cells if cell.kind == "markdown")
    assert len(re.findall(r"^# ", markdown, flags=re.MULTILINE)) == 1


@every_tutorial
def test_a_tutorials_markdown_attaches_nothing(path: pathlib.Path) -> None:
    # The site's build refuses raw HTML and images (test_docs.py builds it).
    for cell in notebooks.read(path):
        if cell.kind == "markdown":
            assert not cell.attachments


@every_tutorial
def test_a_tutorial_links_to_tutorials_and_public_urls_alone(path: pathlib.Path) -> None:
    names = {tutorial.name for tutorial in TUTORIALS}
    for cell in notebooks.read(path):
        if cell.kind == "markdown":
            for target in re.findall(r"\]\(([^)\s]+)", notebooks.prose(cell.source)):
                # By file name alone: the site's heading anchors and Jupyter's differ.
                assert target.startswith("https://") or target in names, target


@every_tutorial
def test_a_tutorials_outputs_are_text(path: pathlib.Path) -> None:
    for cell in notebooks.read(path):
        for output in cell.outputs:
            assert output["output_type"] in ("stream", "execute_result"), output["output_type"]
            # What the cell printed: a warning or a progress bar on stderr is not the tutorial's.
            assert output.get("name", "stdout") == "stdout"
            assert set(output.get("data", {})) <= {"text/plain"}


@every_tutorial
def test_a_tutorials_parameters_are_literals(path: pathlib.Path) -> None:
    assert notebooks.parameters(notebooks.read(path))


@every_tutorial
def test_a_tutorial_installs_the_trees_sdk(path: pathlib.Path) -> None:
    source = "\n".join(cell.source for cell in notebooks.read(path) if cell.kind == "code")
    pins = re.findall(r"goaly-looper==([^\s\"']+)", source)
    # A 1xx notebook needs the SDK alone, so it installs it; a later one runs from a checkout.
    assert pins == ([__version__] if path.name.startswith("1") else [])


@every_tutorial
def test_a_tutorial_takes_the_key_from_its_placeholder(path: pathlib.Path) -> None:
    cells = notebooks.read(path)
    source = "\n".join(cell.source for cell in cells if cell.kind == "code")
    sessions = source.count("ServiceClient.Config(")
    # The reader pastes the key once, in the settings cell, and every session sends it.
    assert source.count("api_key=API_KEY") == sessions
    assert notebooks.parameters(cells).get("API_KEY") == ("YOUR_API_KEY" if sessions else None)


@every_tutorial
def test_the_index_lists_a_tutorial(path: pathlib.Path) -> None:
    assert f"({path.name})" in (notebooks.TUTORIALS / "index.md").read_text()


def test_a_cells_python_skips_its_magics() -> None:
    source = '%pip install -q "goaly-looper"\n!ls\nx = 1\n    %time y = 2'
    assert notebooks.python(source) == "\n\nx = 1\n"


def test_prose_leaves_code_out() -> None:
    source = "# Title\n\n```text\n# not a title <b>\n```\n\nUse `<think>` here."
    assert notebooks.prose(source) == "# Title\n\n\n\nUse  here."


def code_cell(*lines: str, tags: tuple[str, ...] = ()) -> dict[str, Any]:
    metadata = {"tags": list(tags)} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": list(lines),
    }


def saved_notebook(path: pathlib.Path, *cells: dict[str, Any]) -> dict[str, Any]:
    title = {"cell_type": "markdown", "metadata": {}, "source": ["# Example"]}
    notebook = {"cells": [title, *cells], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    for number, cell in enumerate(notebook["cells"]):
        cell["id"] = f"cell-{number}"
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return notebook


async def test_record_saves_what_each_code_cell_printed(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "999_example.ipynb"
    saved = saved_notebook(
        path,
        code_cell('PROJECT = "example"', tags=("parameters",)),
        code_cell("import asyncio\n", "await asyncio.sleep(0)\n", "print(PROJECT)\nprint('done')"),
        code_cell("quiet = True"),
    )
    await notebooks.record(path, {"PROJECT": "dev"})
    expected = copy.deepcopy(saved)
    for count, cell in enumerate(expected["cells"][1:], start=1):
        cell["execution_count"] = count
    stream = {"name": "stdout", "output_type": "stream", "text": ["dev\n", "done\n"]}
    expected["cells"][2]["outputs"] = [stream]
    assert json.loads(path.read_text(encoding="utf-8")) == expected


async def test_record_writes_nothing_when_a_cell_fails(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "999_example.ipynb"
    saved_notebook(
        path,
        code_cell("X = 1", tags=("parameters",)),
        code_cell("print('printed before the failure')"),
        code_cell("raise RuntimeError('the service refused')"),
    )
    before = path.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="the service refused"):
        await notebooks.record(path, {})
    assert path.read_text(encoding="utf-8") == before
