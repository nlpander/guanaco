test-guanaco

To run:
 * Update the *.toml file.
 * Certain params can be overridden with flags.


```
$ python proc_generate_philosophy_debate.py --help

       USAGE: proc_generate_philosophy_debate.py [flags]
flags:

proc_generate_philosophy_debate.py:
  --config: Path to the TOML config
    (default: 'FredRalph_p1.toml')
  --decay-constant: decay constant for the exponential decay
    (default: '0.05')
    (a number)
  --ggml-model: Path to the GGML model
  --[no]gradio: Whether to spin up a UI or not
    (default: 'false')
  --max-temp-randomness: Max temperature randomness from baseline
    (default: '0.0')
    (a number)
  --output: Output file
  --period: number of periods of exponential decay required in the conversation
    (a number)
  --rounds: Number of rounds to execute
    (an integer)
  --temperature-mode: The style of decay or randomness you desire for the conversation
    (default: 'none')
```

For example:
```
$ python proc_generate_philosophy_debate.py \
      --ggml-model /home/fabian/dev/llama.cpp/models/7B/ggml-model-f16-q4_0.bin \
      --rounds 1 \
      --out put output.txt \
      --config cfgs/WarrenRay_p3.toml \
      --gradio
```