# looper_cookbook/tests/e2e/

The cookbook's tests against a real Looper deployment. Each runs the real
SDK against the API root that `$LOOPER_URL` names, with your key in
`LOOPER_API_KEY`, and skips when `LOOPER_URL` is not set. Each run spends
that deployment's GPUs and your organization's credit.

| File | What it shows |
|---|---|
| `test_agent_session.py` | `AgentSession` sessions on a sampler of the base model, which trains nothing: a one-turn answer; a task whose answer is only in a tool's reply, so the right answer shows the model read the prompt `continue_prompt` built; and a group of attempts at once, each in a sandbox of its own |
| `test_train_rl.py` | The RL loop on a LoRA run of Qwen3.5-9B, trained by two calls: each step trains on the step before it, an evaluation samples the weights just published, the first report is the baseline, a save lands where `save_every` puts it, and a second call goes on from the run's step. It makes no claim that the model learns |
| `test_train_sft.py` | The SFT loop on a LoRA run of Qwen3.5-9B, trained by two calls on the bundled conversations: each step reports the targets and tokens of `dataset.batch(k)`, at the rate the schedule names for it, and a second call goes on from the run's step, into the dataset's second pass. It makes no claim that the model learns |
| `test_train_rl_async.py` | The asynchronous RL loop on a LoRA run of Qwen3.5-9B at `max_policy_lag=1`: the loop starts two steps' groups at once, the second step trains groups sampled before the first step's weights were published, no step trains a group older than the bound, and the baseline and the evaluations fall where `train_rl`'s do. It makes no claim that the model learns or that the overlap saves time |
| `test_code_rl.py` | The `code_rl` recipe's own `main` on a few of DeepCoder's problems: two steps of sixteen episodes on a LoRA run of Qwen3.5-9B, and the held-out problems before and after. The model's programs run in Docker containers on this machine, so it also needs `LOOPER_TEST_DOCKER=1` |

```bash
LOOPER_URL=https://api.looper.goaly.ai pytest -q -o asyncio_mode=auto -s \
    looper_cookbook/tests/e2e/test_agent_session.py
```
