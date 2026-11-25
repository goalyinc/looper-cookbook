# Code RL: DeepCoder's programming problems

This recipe uses reinforcement learning to train a model to solve competitive programming problems. Each problem asks for a Python program, and the model gets a reward of 1 if its program passes the problem's tests, and 0 if not.

It trains on the problems of [DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) (Luo et al., 2025), with LiveCodeBench's prompt, and gives the model a tool to test its program before it answers.

It's built only from the cookbook's own parts, so it's also a complete example of how they fit together:

- **An [episode session](../../../docs/episode-sessions.md)** runs each attempt, turn by turn.
- **An [environment](../../../docs/environments.md)** gives the model its testing tool.
- **A [verifier](../../../docs/verifiers.md)** scores the final program.
- **The [RL loop](../../../docs/reinforcement-learning.md)** trains on the scores.

This page covers:

- [An attempt at a problem](#an-attempt-at-a-problem): what happens from the question to the score.
- [Run it](#run-it): the commands, the flags, and what the recipe prints.
- [The data](#the-data): the problems and tests it trains on.
- [What is not here yet](#what-is-not-here-yet): known gaps.

Words used on this page:

| Word | Meaning |
|---|---|
| **Task** | One programming problem, with the verifier that scores answers to it. |
| **Attempt** | One try at a problem: the model's turns, its final program, and the reward. Also called an **episode**. |
| **Group** | All the attempts at the same problem in one training step. The model learns by comparing their rewards. |
| **Test** | An input for the program, and the output it must print or return. |
| **Sandbox** | An isolated place where the model's programs run, here a Docker container on your machine. |
| **Held-out problems** | Problems the run never trains on, used to measure the model. |

## An attempt at a problem

Each attempt is an `AgentSession` of at most two turns:

1. **The question.** The model gets the problem's statement, in LiveCodeBench's prompt, which asks for a Python program in a code block.
2. **An optional check.** The model can call a tool, `check_solution`, with a draft of its program. A `CodingEnvironment` runs the draft on the problem's first two tests, in a sandbox, and tells the model how many passed. For the first test that failed, it also shows the input, the expected output, what the program printed, and any error. The check earns no reward.
3. **The answer.** The model's last message, with its program in a code block.
4. **The score.** A `CodeTestsVerifier` runs that program against every test the problem keeps, in a separate sandbox the model never worked in. The reward is 1 if it passes them all, and 0 if not.

So the model can see two of the tests while it works, but the tests that decide its reward stay hidden in the verifier.

`train_rl.train` runs each problem in a batch `--group-size` times, and trains each problem's attempts as one group, with `Grpo`. Before the first step and after the last, the recipe also tries DeepCoder's test problems, which it never trains on, once each with the model's most likely answer. The share it solves is the recipe's measurement.

## Run it

You need a Looper API key in `LOOPER_API_KEY`, and Docker on the machine that runs the recipe. The model's programs run in containers on your machine, not on Looper.

To run it with its defaults:

```bash
python -m looper_cookbook.recipes.code_rl.train
```

For a quick trial that takes minutes and downloads little:

```bash
python -m looper_cookbook.recipes.code_rl.train --problems 32 --eval-problems 16 --steps 4 --batch-size 8
```

| Flag | Default | What it sets |
|---|---|---|
| `--model` | `Qwen/Qwen3.5-9B` | The model to train, as a LoRA with the rank set by `--lora-rank` (32 by default). `--full` trains every parameter instead. |
| `--problems` | all | How many training problems to read from each subset. With a number, the problems are streamed. Without one, every training subset is downloaded in full, about 5.4 GB of files. |
| `--eval-problems` | 64 | How many held-out problems to use from each test subset. 0 measures nothing. |
| `--steps`, `--batch-size`, `--group-size` | 100, 16, 8 | The number of optimizer steps, problems per step, and attempts per problem. |
| `--max-turns` | 2 | The most turns in an attempt. |
| `--max-turn-tokens` | 2048 | The token limit for one turn, which is the most tokens the sampler returns for it. |
| `--max-episode-tokens` | 8192 | The token limit for a whole attempt, counting its prompt, every turn and every tool reply together. |
| `--public-tests`, `--max-tests` | 2, 32 | How many tests `check_solution` runs, and how many tests a problem keeps. |
| `--sandbox`, `--image` | `docker`, `python:3.12-slim` | Where programs run. `--sandbox local` runs them as you, with no isolation, so only use it for a model you trust. |
| `--eval-every`, `--save-every` | none | How many steps between held-out measurements, and between saves. |

The recipe prints one line per report, with the step and then whichever of these metrics the report has. The first line measures the held-out problems before any training.

| Metric | What it is |
|---|---|
| `reward` | The share of the step's attempts whose program passed every test. |
| `groups_trained` | The problems whose attempts didn't all score the same. Only those train, because if every attempt scored the same, there's nothing to compare. |
| `turns` | The average number of turns per attempt. Above 1 means some attempts called `check_solution`. |
| `stop/max_turn_tokens` | The share of attempts ended by the token limit for one turn. These attempts score 0. |
| `stop/max_episode_tokens` | The share of attempts ended by the token limit for the whole attempt. |
| `eval/reward`, `eval/reward_stderr` | The share of held-out problems solved, and its standard error. |
| `eval/stop/…` | The same two stop shares, for the held-out attempts. |

**Each attempt starts its own sandboxes.** A step of 16 problems with 8 attempts each starts up to 256 of them, so the machine that runs the recipe needs to run that many containers at once.

## The data

The problems come from [`agentica-org/DeepCoder-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) (MIT license), fixed at its version of 2025-04-09. The recipe trains on the dataset's train splits, and measures on its test splits:

| Subset | Used for | Problems | What the tests check |
|---|---|---|---|
| `primeintellect` | Training | 16,252 | Whole programs that read standard input and write standard output. |
| `taco` | Training | 7,436 | Whole programs, and functions called with arguments. |
| `lcbv5` | Training, and held-out | 599 for training, 279 held out | Whole programs, and LeetCode functions with starter code. |
| `codeforces` | Held-out | 408 | Whole programs, usually one test each. |

- **A problem keeps at most 32 tests,** which are its first 32 tests of at most 65,536 characters each. Some problems have hundreds of megabytes of tests. And a test whose expected output is longer than the 1 MiB a sandbox keeps could never pass. Leaving out the largest tests means a program too slow for them can still pass.
- **Common imports are added for you.** A function's solution can use `List` or `defaultdict` without importing them, as LeetCode's do, because the recipe writes LiveCodeBench's imports before every program.

## What is not here yet

- **A pass/fail check and a run log.** The recipe prints its reports, but doesn't check whether the model learned, or write a log of the run.
- **Problems with several right answers** can't be scored by comparing outputs. That would need a checker for each problem, which the dataset doesn't have.
- **A solution that starts with a `from __future__` import** fails, because the added imports come before it.
- **A pool of ready sandboxes.** Every attempt starts its own containers.
