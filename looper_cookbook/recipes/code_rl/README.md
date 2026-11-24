# Code RL: DeepCoder's programming problems

This recipe trains a model with reinforcement learning to solve competitive programming problems. It trains on the problems of [DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) (Luo et al., 2025), under LiveCodeBench's prompt, and gives the model a tool it may call to check a solution before it answers. It is written with the cookbook's own parts: an episode session over an environment, and a verifier that scores it.

## An attempt at a problem

Each attempt is an `AgentSession` of at most two turns:

1. **The question.** The problem's statement under LiveCodeBench's prompt, which asks for a Python program in a code block.
2. **An optional check.** The model can call a tool, `check_solution`, with a draft of its program. A `CodingEnvironment` runs the draft on the problem's first two tests, in a sandbox, and tells the model how many passed. For the first test that failed, it also shows the input, the expected output, what the program printed, and any error. The check earns no reward.
3. **The answer.** The model's last message, with its program in a code block.
4. **The score.** A `CodeTestsVerifier` runs that program against every test the problem keeps, in a sandbox of its own that the model never acted in. The reward is 1 when it passes them all, else 0.

`train_rl.train` attempts every problem of a batch `--group-size` times and trains each problem's attempts as one group under `Grpo`. Problems the run never trains, DeepCoder's test subsets, are attempted greedily before the first step and after the last: the share solved is the recipe's measurement.

## Run it

You need a Looper API key in `LOOPER_API_KEY`, and Docker on the machine that runs the recipe: the model's programs run in containers there, not on Looper.

```bash
python -m looper_cookbook.recipes.code_rl.train
```

A trial that takes minutes and downloads little:

```bash
python -m looper_cookbook.recipes.code_rl.train --problems 32 --eval-problems 16 --steps 4 --batch-size 8
```

| Flag | Default | What it sets |
|---|---|---|
| `--model` | `Qwen/Qwen3.5-9B` | The run's model, a LoRA of `--lora-rank` 32. `--full` trains every parameter |
| `--problems` | all | Train problems read of each subset. With a number, the subset is streamed; without, every train subset is downloaded whole, about 5.4 GB of files |
| `--eval-problems` | 64 | Held-out problems of each test subset. 0 measures nothing |
| `--steps`, `--batch-size`, `--group-size` | 100, 16, 8 | Optimizer steps, problems a step, attempts a problem |
| `--max-turns` | 2 | An attempt's turns |
| `--max-turn-tokens` | 2048 | A turn's token budget: the most tokens the sampler returns for one turn |
| `--max-episode-tokens` | 8192 | An attempt's token budget: its prompt, every turn and every tool reply together. Qwen3.5-9B's sampler for RL holds 8,192 tokens |
| `--public-tests`, `--max-tests` | 2, 32 | The tests `check_solution` runs, and the tests a problem keeps |
| `--sandbox`, `--image` | `docker`, `python:3.12-slim` | Where programs run. `--sandbox local` runs them as you, with no isolation: use it only for a model you trust |
| `--eval-every`, `--save-every` | none | Steps between held-out measurements, and between saves |

The recipe prints a line per report: its step, then the metrics below that the report holds. The first line is the held-out problems before any training.

| Metric | What it is |
|---|---|
| `reward` | The share of the step's attempts whose program passed every test |
| `groups_trained` | The problems whose attempts did not all score the same. Only those train |
| `turns` | The turns an attempt took, as a mean. Above 1, some attempts called `check_solution` |
| `stop/max_turn_tokens` | The share of attempts that a turn's token budget ended. Such an attempt scores 0 |
| `stop/max_episode_tokens` | The share of attempts that the attempt's token budget ended |
| `eval/reward`, `eval/reward_stderr` | The share of the held-out problems solved, and its standard error |
| `eval/stop/…` | The two stop shares, of the held-out attempts |

On a 192-core machine, 64 Docker sandboxes at once each started, ran a program on 16 tests and closed in 7 s. A step of 16 problems and 8 attempts starts up to 256.

## The data

[`agentica-org/DeepCoder-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) (MIT), pinned at its revision of 2025-04-09. The recipe trains on its train splits and measures on its test splits:

| Subset | Split | Problems | Tests |
|---|---|---|---|
| `primeintellect` | train | 16,252 | Programs: standard input to standard output |
| `taco` | train | 7,436 | Programs, and functions called with arguments |
| `lcbv5` | train, test | 599, 279 | Programs, and LeetCode functions with starter code |
| `codeforces` | test | 408 | Programs, usually one test each |

A problem keeps its first 32 tests among those of at most 65,536 characters. Some problems carry hundreds of megabytes of tests, and a test whose expected output is longer than the 1 MiB a sandbox keeps could never pass. Leaving out the largest tests lets a program too slow for them pass.

A function's solution may use `List` or `defaultdict` without importing them, as LeetCode's do: the recipe writes LiveCodeBench's imports before every program.

## Checked against the dataset's own solutions

`taco` and `primeintellect` carry solutions. If the recipe reads a problem's tests right and the check runs them right, a problem's own solution passes. On the first 150 problems of each, in Docker (`python -m looper_cookbook.tests.code_rl_solutions --problems 150`, 2026-11-18):

| Subset | Tests | Solutions that pass |
|---|---|---|
| `taco` | programs | 129 of 131 (98.5%) |
| `taco` | functions | 19 of 19 (100%) |
| `primeintellect` | programs | 123 of 150 (82.0%) |

- **The two `taco` failures** are a solution that opens with a `from __future__` import, which cannot follow the prelude, and one that does not parse.
- **Most `primeintellect` failures** are problems with more than one right answer, such as "print any valid order". The dataset holds one expected output, so a different right answer fails. On those problems no program can be told right from wrong.

## Results

**No run long enough to show learning is recorded here yet.** One short run shows the recipe working end to end, against Looper's dev service on 2026-11-19, with the model's programs in Docker containers (`tests/e2e/test_code_rl.py`: 8 problems of each train subset, 4 of each test subset, two steps of 4 problems and 4 attempts, with a turn's budget of 1,024 tokens and an attempt's of 8,192):

| | Held-out problems solved | Train reward | Turns an attempt | Train attempts cut off |
|---|---|---|---|---|
| Before training | 3 of 8 | | | |
| Step 1 | | 0.125 | 1.44 | 56% |
| Step 2 | 4 of 8 | 0.250 | 1.25 | 75% |

- **The model called the tool in 44% and 25% of its attempts,** and answered at once in the rest.
- **A turn's budget cut off most attempts, and the attempt's budget none.** Every cut-off attempt ended as `max_turn_tokens`, at 1,024 tokens a turn, and none reached its 8,192. Of the held-out attempts, 5 of 8 were cut off before training and 4 of 8 after. A cut-off attempt scores 0. Why turns run that long is not checked: the run saved no turn's text. The recipe's default is 2,048 tokens a turn, and the share cut off there is not measured yet. It is the first number to read in a longer run: the recipe prints it each step, with the share each budget ended.
- **One more held-out problem solved is not learning.** With 8 problems the share's standard error is 0.19, and an earlier run of the same test solved 3 of 8 before and after.

## What is not here yet

- **A gate and a run log.** The recipe prints its reports. A learning gate and the files a run writes come with the cookbook's other recipes.
- **Problems with several right answers** cannot be scored by comparing outputs. A checker per problem would need the dataset to carry one.
- **A solution that opens with a `from __future__` import** fails, since the prelude is written before it.
- **A pool of warm sandboxes.** Every attempt starts its own containers.
