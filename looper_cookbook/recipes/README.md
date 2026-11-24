# recipes/

The cookbook's entry points: scripts with command-line
flags, run as modules once the cookbook is installed. Each defaults to
`https://api.looper.goaly.ai`. Override `--url` to use another API root,
without `/v1`, such as dev's, `https://dev.api.looper.goaly.ai`.

None has a gate yet. `rl_basic.py` measures held-out problems before and
after it trains.

## Usage

Run a recipe from the repository's root, with the cookbook installed. Each needs a
Looper server and your API key, and spends that server's GPUs, so no output
is shown here: "What to expect" below says what each prints.

```bash
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.sft_basic
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.sft_basic --data my.jsonl --epochs 2
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.rl_basic --steps 20
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.code_rl.train --problems 32 --eval-problems 16 --steps 4 --batch-size 8
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.sampling_basic --checkpoint sft-basic-1a2b3c4d
```

`code_rl` also needs Docker on this machine, where the model's programs run.
`--help` lists a recipe's flags, and needs no server.

`--model` names the run's model, and the recipe gets that model's formatter
from `looper_cookbook.message_formatters.models`. `sft_basic.py` and
`rl_basic.py` take any model listed there, as in
`--model meta-models/Muse-Glimmer-30B`. `--thinking` turns a Qwen model's
thinking mode on; Muse Glimmer has no such mode.

| File | What it does |
|---|---|
| `sft_basic.py` | An SFT job from its data to its checkpoint. The model's message formatter makes one datum per assistant turn, written as the model's own chat template writes it, and `looper_cookbook.train_sft` trains them, a step per batch of `--batch-size` over `--epochs` passes, submitting ahead and decaying its rate. Its data is GSM8K's train split, or with `--data` a JSONL file of conversations. A LoRA of `--lora-rank` 32 by default, every parameter with `--full` |
| `rl_basic.py` | An RL job from its problems to its checkpoint. Each problem is a task: the question asked through the model's message formatter, with an instruction to end on `#### <number>`, and a `FinalNumber` verifier that gives 1 when that number is the answer. The run's training client creates its sampler, and `looper_cookbook.train_rl` takes `--steps` steps: `--group-size` attempts at each of `--batch-size` problems, a `Grpo` step, a publish. It measures held-out problems, answered greedily, before its first step, after its last and every `--eval-every` steps. Its problems are GSM8K's train split, measured on its test split, or with `--data` and `--eval-data` JSONL files of `{"question", "answer"}` lines. A LoRA of `--lora-rank` 32 by default, every parameter with `--full` |
| [code_rl/](code_rl/README.md) | RL on DeepCoder's programming problems: `python -m looper_cookbook.recipes.code_rl.train`. `tasks.py` reads the dataset's four subsets as tasks: an `AgentSession` of two turns over a `CodingEnvironment`, whose `check_solution` runs a problem's first two tests, and a `CodeTestsVerifier` that scores the final program on every test the problem keeps. Programs run in Docker sandboxes on the machine that runs the recipe. It measures DeepCoder's test subsets before and after |
| `sampling_basic.py` | A standalone sampler of Qwen3.5-9B's base, of Qwen3.8-27B's with `--model`, or of a checkpoint in the project with `--checkpoint NAME`: every question, a grade-school math problem as `sft_basic.py`'s default data is, sent at once for `--n` completions, each prompted through the model's message formatter with thinking off, as `sft_basic.py` trains. The results are printed as they all arrive, with each request's token counts and the sampler's totals, then the sampler is stopped |

## The default data

GSM8K ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), MIT)
is 8,792 grade-school math problems, each with a worked answer that ends
in `#### <number>`. The recipe trains its train split, 7,473 problems. The test
split's 1,319 are left out.

- **What the recipe reads.** The `main` configuration's train split, which
  `datasets` downloads into Hugging Face's cache at the revision
  `sft_basic.py` pins.
- **What a record becomes.** A one-turn conversation: the question,
  answered by the worked answer as the dataset has it, its `<<48/2=24>>`
  calculator notes and its `#### <number>` line included.
- **Its size.** Under Qwen3.5-9B's chat template it is 7,473 datums of
  1,467,246 tokens, 932,392 of them targets. A datum is 196 tokens on
  average and 549 at most. One pass in batches of 64 is 117 steps.

Your own file takes its place with `--data`: a JSONL file, one conversation
a line ([example_data/](../example_data/README.md)).

`rl_basic` asks the same train split's questions and rewards the number its
worked answer ends on. Its `--data` is a JSONL file of problems, each a
`question` and its `answer`, a number.

## A quick run

The bundled conversations train in 8 steps and need no dataset download,
and so do the bundled math problems:

```bash
python -m looper_cookbook.recipes.sft_basic --data looper_cookbook/example_data/conversations.jsonl --batch-size 8 --epochs 4
python -m looper_cookbook.recipes.rl_basic --data looper_cookbook/example_data/math_problems.jsonl --steps 4 --batch-size 8
```

`--model`, `--full`, `--lora-rank` and `--max-seq-len` pick what to train;
`rl_basic`'s sizes come from its prompts and `--max-turn-tokens`, and its
sampler's from those and `--concurrency`.

## What to expect

- **`sft_basic`** prints what its first datum trains, a line per step, the
  loss over the run's first and last tenth, then its checkpoint.
  - **On the bundled conversations,** with `--batch-size 8 --epochs 4
    --max-seq-len 2048` (2026-11-10, a rank-32 LoRA of
    Qwen3.5-9B): 18 datums of 1,366 tokens, 805 of them targets, and 12
    steps. The loss was 2.09 at step 1 and 0.06 to 0.14 over the last
    three, and the checkpoint was 1.14 GB. The run was the same data and
    loop under the recipe's earlier name, `sft_chat`.
- **`rl_basic`** prints how its first problem is asked, a line per step
  with its reward and the step its completions came from, the reward over
  the run's first and last tenth, then its checkpoint.
  - **On GSM8K,** with `--steps 12 --batch-size 16` (2026-11-12, a
    rank-32 LoRA of Qwen3.5-9B, its defaults otherwise): the reward was 0.82 at step 1, 0.85 over steps
    1 to 4 and 0.95 over steps 9 to 12. A step took 30 s, and the run
    with its sampler 3 min to start. Without thinking, the base model's
    first step already scored 0.82, so the reward has little room to
    rise.
- **`sampling_basic`** prints each question with its token counts and its
  completions, their finish reasons and mean log-probabilities, then the
  sampler's total token counts.
