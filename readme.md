test-guanaco

To run:
 * Update the *.toml file.
 * Certain params can be overridden with flags.


```
python proc_generate_philosophy_debate.py --help

       USAGE: proc_generate_philosophy_debate.py [flags]
flags:

proc_generate_philosophy_debate.py:
  --config: Path to the TOML config
    (default: 'FredRalph_p1.toml')
  --ggml-model: Path to the GGML model
  --output: Output file
  --rounds: Number of rounds to execute
    (an integer)
```

For example:
```
$ python proc_generate_philosophy_debate.py \
    --ggml-model /home/fabian/dev/llama.cpp/models/7B/ggml-model-f16-q4_0.bin \
    --rounds 2 \
    --output output.txt
```