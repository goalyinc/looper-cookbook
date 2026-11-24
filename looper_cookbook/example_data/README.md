# example_data/

Example data for the recipes, small enough to read.

## Usage

Give a file to its recipe with `--data`, from the repository's root. Each
run needs a Looper server:

```bash
python -m looper_cookbook.recipes.sft_basic --data looper_cookbook/example_data/conversations.jsonl --batch-size 8 --epochs 4
python -m looper_cookbook.recipes.rl_basic --data looper_cookbook/example_data/math_problems.jsonl --steps 4 --batch-size 8
```

| File | What it holds |
|---|---|
| `conversations.jsonl` | 16 conversations for `recipes/sft_basic.py --data`, one per line as `{"messages": [...]}` of chat messages: each asks about an animal and is answered with a three-line card, `Habitat`, `Diet` and `Fun fact`. Two have a follow-up question, so they train two turns, and four open with a system message |
| `math_problems.jsonl` | 32 grade-school math problems for `recipes/rl_basic.py --data` and the `302_rl_loop` tutorial, one per line as `{"question": ..., "answer": ...}`: the first 32 of GSM8K's train split ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) at `740312ad`, MIT), each answer the number its worked answer ends on, without thousands commas |

Your own file has the same shape. A conversation's line may also carry
`tools`, the JSON schemas of the tools its conversation calls.
