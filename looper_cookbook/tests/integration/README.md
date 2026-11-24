# looper_cookbook/tests/integration/

The cookbook's integration tests: each crosses one real boundary that is
not a Looper server.

| Directory | Boundary |
|---|---|
| `sandboxes/` | Real processes: a local sandbox's commands, with their input, their deadline and their process group, and what the process runner keeps of an output. With `LOOPER_TEST_DOCKER=1`, real Docker containers too: what a container is denied, and what is left when it closes |
| `environments/` | The example environments over a real `bash` and a real Python, in a local sandbox. With `LOOPER_TEST_DOCKER=1`, in real Docker containers too |
| `code_rl/` | DeepCoder's problems over real processes: a program test, a LeetCode function with its starter code and a TACO function, each passed by its solution, and a whole episode of the recipe's task, checked on its public tests and scored on them all. The rows are the dataset's own, cut to a few tests |
| `episode_sessions/` | Whole episodes of turns written by hand, over real processes: the agent session's tool calls in a real local sandbox, and a verifier's hidden tests in another |

```bash
pytest -q -o asyncio_mode=auto looper_cookbook/tests/integration
LOOPER_TEST_DOCKER=1 pytest -q -o asyncio_mode=auto looper_cookbook/tests/integration
```

The second command starts containers of `python:3.12-slim` on this
machine's Docker daemon and removes each. The `episode_sessions/` tests use
local sandboxes alone.
