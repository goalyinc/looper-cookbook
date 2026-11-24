"""An environment and a message formatter fit each other with nothing between
them.

The formatter lists the environment's tools in a prompt and reads a sampled
turn's call back with their types; the call is an action, and the
observation's text is the tool message of the next prompt. ``AgentSession``
does just this, turn after turn.
"""

from __future__ import annotations

from looper_cookbook.environments.base import Action
from looper_cookbook.environments.bash import BashEnvironment
from looper_cookbook.message_formatters.base import call_arguments
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter
from looper_cookbook.message_formatters.unit_tests import tokenizers
from looper_cookbook.sandboxes.base import CommandResult
from looper_cookbook.sandboxes.unit_tests.scripted import Prebuilt, ScriptedSandbox

SAMPLED = (
    "<tool_call>\n<function=bash>\n<parameter=command>\necho hi\n</parameter>\n"
    "</function>\n</tool_call>"
)


async def test_a_sampled_call_runs_and_its_reply_is_the_next_prompts_tool_message() -> None:
    tokenizer = tokenizers.qwen()
    formatter = Qwen3_5MessageFormatter(tokenizer, enable_thinking=False)
    sandbox = ScriptedSandbox(CommandResult(exit_code=0, stdout="hi\n"))
    environment = await BashEnvironment.Config(sandbox=Prebuilt(sandbox)).build()
    messages = [{"role": "user", "content": "Say hi from the shell."}]

    # The prompt lists the environment's tools as the environment declares them.
    assert '{"type": "function", "function": {"name": "bash", ' in formatter.format_prompt(
        messages, environment.tools
    )

    sampled = [*tokenizer(SAMPLED, add_special_tokens=False)["input_ids"], formatter.end_of_turn_id]
    reply, ended = formatter.parse_response(sampled, environment.tools)
    assert ended
    (call,) = reply["tool_calls"]
    name, arguments = call_arguments(call)
    observation = await environment.step(Action(name=name, arguments=arguments))
    assert observation.content == "hi\n"
    assert sandbox.commands == [("bash -c 'echo hi' 2>&1", 60.0, None)]

    messages += [reply, {"role": "tool", "name": name, "content": observation.content}]
    assert formatter.format_prompt(messages, environment.tools).endswith(
        "<|im_start|>user\n<tool_response>\nhi\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
