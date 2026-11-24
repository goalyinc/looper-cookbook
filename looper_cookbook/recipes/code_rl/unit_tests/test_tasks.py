"""DeepCoder's rows as problems and tasks: each subset's test format, the
tests a problem keeps, the prompt, and what a task gives the tool and the
verifier (looper_cookbook/recipes/code_rl/tasks.py). The rows are the
dataset's own, cut to a few tests."""

from __future__ import annotations

import json

from looper_cookbook.environments.coding import CodeTest, CodingEnvironment
from looper_cookbook.recipes.code_rl import tasks
from looper_cookbook.sandboxes.local import LocalSandbox
from looper_cookbook.verifiers.code_tests import CodeTestsVerifier

CODEFORCES = '[{"input": "2\\n1 9\\n8 4", "output": "1 9\\n4 8"}]'
PRIME_INTELLECT = (
    '[{"type": "stdin_stdout", "input": "1 1\\n", "output": "0\\n"}, '
    '{"type": "stdin_stdout", "input": "2 2\\n", "output": "8\\n"}]'
)
LCB_STDIN = '[{"input": "1\\nabc\\n", "output": "YES\\n", "testtype": "stdin"}]'
LCB_FUNCTIONAL = (
    '[{"input": "[\\"7868190130M7522\\", \\"5303914400F9211\\"]", "output": "2", '
    '"testtype": "functional"}, {"input": "5\\n[[2, 4], [0, 2]]", "output": "[3, 2]", '
    '"testtype": "functional"}]'
)
TACO_FUNCTION = {
    "fn_name": "is_anagram",
    "inputs": [["foefet", "toffee"], ["apple", "pale"]],
    "outputs": [[True], [False]],
}


def test_a_list_of_inputs_and_outputs_is_program_tests() -> None:
    expected = [CodeTest(input="2\n1 9\n8 4", output="1 9\n4 8")]
    assert tasks.code_tests(CODEFORCES) == expected
    # Read already, as well as JSON text.
    assert tasks.code_tests(json.loads(CODEFORCES)) == expected
    assert tasks.code_tests(PRIME_INTELLECT) == [
        CodeTest(input="1 1\n", output="0\n"),
        CodeTest(input="2 2\n", output="8\n"),
    ]
    assert tasks.code_tests(LCB_STDIN, {"func_name": None}) == [
        CodeTest(input="1\nabc\n", output="YES\n")
    ]


def test_a_functional_test_calls_the_function_its_rows_metadata_names() -> None:
    assert tasks.code_tests(LCB_FUNCTIONAL, {"func_name": "countSeniors"}) == [
        CodeTest(
            input='["7868190130M7522", "5303914400F9211"]', output="2", function="countSeniors"
        ),
        # Two arguments, a JSON value a line.
        CodeTest(input="5\n[[2, 4], [0, 2]]", output="[3, 2]", function="countSeniors"),
    ]
    # With no name there is nothing to call, so the tests are left out.
    assert tasks.code_tests(LCB_FUNCTIONAL, {"func_name": None}) == []


def test_tacos_calls_give_their_arguments_as_a_list_and_their_result_in_one() -> None:
    assert tasks.code_tests(TACO_FUNCTION) == [
        CodeTest(input='"foefet"\n"toffee"', output="true", function="is_anagram"),
        CodeTest(input='"apple"\n"pale"', output="false", function="is_anagram"),
    ]
    # A result that is itself a list stays one.
    listed = {"fn_name": "pair", "inputs": [[1]], "outputs": [[[1, 2]]]}
    assert tasks.code_tests(listed) == [CodeTest(input="1", output="[1, 2]", function="pair")]


def test_tacos_programs_read_text_or_a_list_of_lines() -> None:
    raw = {"inputs": ["5 3\n1 2\n", ["5", "4 1"]], "outputs": ["4\n", ["2", "8"]]}
    assert tasks.code_tests(json.dumps(raw)) == [
        CodeTest(input="5 3\n1 2\n", output="4\n"),
        CodeTest(input="5\n4 1\n", output="2\n8\n"),
    ]


def test_a_problem_keeps_its_first_tests_that_are_not_too_long() -> None:
    raw = [
        {"input": "1", "output": "1"},
        {"input": "x" * 50, "output": "2"},
        {"input": "3", "output": "3"},
        {"input": "4", "output": "4"},
    ]
    row = {"problem": "Echo the number.", "tests": json.dumps(raw)}
    made = tasks.problem(row, max_tests=2, max_test_chars=20)
    assert made is not None
    assert made.tests == (CodeTest(input="1", output="1"), CodeTest(input="3", output="3"))
    assert made.question == tasks.question("Echo the number.")


def test_a_row_with_no_statement_or_no_test_it_can_keep_is_no_problem() -> None:
    assert (
        tasks.problem({"problem": " ", "tests": CODEFORCES}, max_tests=4, max_test_chars=99) is None
    )
    assert tasks.problem({"problem": "p", "tests": "[]"}, max_tests=4, max_test_chars=99) is None
    assert (
        tasks.problem({"problem": "p", "tests": CODEFORCES}, max_tests=4, max_test_chars=3) is None
    )
    # An lcbv5 row's metadata may come as JSON text.
    row = {"problem": "p", "tests": LCB_FUNCTIONAL, "metadata": '{"func_name": "countSeniors"}'}
    made = tasks.problem(row, max_tests=1, max_test_chars=99)
    assert made is not None and made.tests[0].function == "countSeniors"


def test_a_question_is_asked_under_live_code_benchs_prompt() -> None:
    assert tasks.question("Add two numbers.\n") == (
        "You are an expert Python programmer. You will be given a question (problem "
        "specification) and will generate a correct Python program that matches the "
        "specification and passes all tests.\n\nAdd two numbers.\n"
        "### Format: Read the inputs from stdin solve the problem and write the answer to "
        "stdout (do not directly test on the sample inputs). Enclose your code within "
        "delimiters as follows. Ensure that when the python program runs, it reads the "
        "inputs, runs the algorithm and writes output to STDOUT.\n"
        "```python\n# YOUR CODE HERE\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
    )
    starter = "class Solution:\n    def add(self, a: int, b: int) -> int:\n        "
    asked = tasks.question("Add two numbers.\n", starter)
    assert "### Format: You will use the following starter code" in asked
    assert f"```python\n{starter}\n```\n\n### Answer:" in asked
    # A blank starter is none.
    assert tasks.question("Add two numbers.\n", "") == tasks.question("Add two numbers.\n")


def test_a_task_shows_the_tool_the_first_tests_and_scores_on_all() -> None:
    made = tasks.problem(
        {
            "problem": "p",
            "tests": json.dumps([{"input": str(n), "output": str(n)} for n in range(5)]),
        },
        max_tests=5,
        max_test_chars=99,
    )
    assert made is not None
    sandbox = LocalSandbox.Config()
    task = tasks.task(
        made, sandbox=sandbox, python="python3.12", public_tests=2, max_episode_tokens=4096
    )
    assert task.messages == [{"role": "user", "content": made.question}]
    assert (task.max_turns, task.max_episode_tokens) == (2, 4096)
    shared = {
        "sandbox": sandbox,
        "timeout_s": 6.0,
        "python": "python3.12",
        "prelude": tasks.LIVE_CODE_BENCH_IMPORTS,
    }
    assert task.environment == CodingEnvironment.Config(tests=made.tests[:2], **shared)
    assert task.verifier == CodeTestsVerifier.Config(tests=made.tests, **shared)


def test_the_subsets_are_the_datasets_train_and_test_splits() -> None:
    assert tasks.SUBSETS == {
        "train": ("primeintellect", "taco", "lcbv5"),
        "test": ("codeforces", "lcbv5"),
    }
