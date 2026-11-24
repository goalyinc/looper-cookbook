# looper_cookbook/tests/

The cookbook's tests beyond its unit tests, which sit beside the sources in
`unit_tests/`. None mocks the SDK: a test that needs a server runs the real
SDK against a real Looper deployment.

## Usage

From the repository's root, after `pip install -e .`:

```bash
pip install pytest pytest-asyncio jinja2
# The unit and integration tests need no server.
pytest -q -o asyncio_mode=auto looper_cookbook
# The integration tests again in real containers, which needs Docker.
LOOPER_TEST_DOCKER=1 pytest -q -o asyncio_mode=auto looper_cookbook/tests/integration
```

The end-to-end tests skip unless `LOOPER_URL` names a Looper API root, and
each run spends that deployment's GPUs and your organization's credit
([e2e/](e2e/README.md)).

| File | What it holds |
|---|---|
| [integration/](integration/README.md) | The sandboxes, the environments and whole episodes of written turns over real processes, and with `LOOPER_TEST_DOCKER=1` in real Docker containers. No server: none of them samples |
| [e2e/](e2e/README.md) | Against a real Looper deployment: the agent session's episodes, sampled from a base model, the RL and SFT loops' contracts on runs of their own, and the `code_rl` recipe at a size that takes minutes |
| `message_formatters.py` | Checks a message formatter against its model's own chat template, with the model's real tokenizer: every prompt's text and tokens, every turn's text, and every turn read back. Run it when a formatter or a template changes: `python -m looper_cookbook.tests.message_formatters qwen3_5 Qwen/Qwen3.5-9B` |
| `code_rl_solutions.py` | Runs DeepCoder's own solutions against its tests as the `code_rl` recipe reads and checks them, in sandboxes: the share that pass shows whether the recipe reads each subset's test format right. `python -m looper_cookbook.tests.code_rl_solutions --problems 150` |
| `notebooks.py` | Reads a tutorial notebook and runs its code cells in order without a Jupyter kernel, with its `parameters` cell's values replaced, and records what each cell printed as its saved output |
