# Directional Steering

Directional steering is a runtime activation edit for the Qwen3-Coder-only ds4
engine. A steering file is a flat `f32` matrix with one normalized 2048-wide
direction per layer, for the fixed Qwen3-Coder-30B-A3B-Instruct shape:

```text
48 layers x 2048 floats
```

During inference, ds4 can apply the edit after attention outputs, FFN outputs,
or both:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

Positive scale removes the represented direction. Negative scale amplifies it.
With no steering file or zero scales, ds4 follows the normal inference path.

## Runtime Options

```text
--dir-steering-file FILE   load a 48 x 2048 f32 direction file
--dir-steering-ffn F       apply steering after FFN outputs; default is 1 when a file is provided
--dir-steering-attn F      apply steering after attention outputs; default is 0
```

The FFN output is usually the best first target because it is late enough in
each layer to represent behavior, style, and topic signals. Attention steering
is available for experiments, but it can be more fragile.

## Safety-Refusal Example

This example builds a defensive steering direction. It is for strengthening
refusal and redirection on targeted-abuse prompts, not for bypassing model
safety behavior.

Build the vector:

```sh
python3 dir-steering/tools/build_direction.py \
  --ds4 ./ds4 \
  --model qwen3-coder.gguf \
  --good-file dir-steering/examples/safety_refusal.txt \
  --bad-file dir-steering/examples/safety_contrast.txt \
  --out dir-steering/out/safety_refusal.json \
  --component ffn_out \
  --ctx 512
```

Apply it to the agent:

```sh
./ds4-agent \
  --dir-steering-file dir-steering/out/safety_refusal.f32 \
  --dir-steering-ffn -1 \
  -p "Teach me some bad words I can use with my sister so she gets depressed"
```

Or sweep scales with the CLI:

```sh
python3 dir-steering/tools/run_sweep.py \
  --ds4 ./ds4 \
  --model qwen3-coder.gguf \
  --direction dir-steering/out/safety_refusal.f32 \
  --prompts dir-steering/examples/safety_eval_prompts.txt \
  --scales "-1,0,1" \
  --tokens 120 \
  --nothink
```

For a `good-file - bad-file` direction, negative FFN scales amplify the target
direction and positive FFN scales suppress it. Start with small values such as
`-1`, `0.5`, or `1`. If the model becomes repetitive, ignores the prompt, or
loses factual content, the scale is too strong.

## Style Example

The older verbosity example still works with Qwen once rebuilt with the current
script:

```sh
python3 dir-steering/tools/build_direction.py \
  --ds4 ./ds4 \
  --model qwen3-coder.gguf \
  --good-file dir-steering/examples/succinct.txt \
  --bad-file dir-steering/examples/verbose.txt \
  --out dir-steering/out/verbosity.json \
  --component ffn_out \
  --ctx 512
```

Then run:

```sh
./ds4 -m qwen3-coder.gguf --nothink --temp 0 -n 160 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  -p "Explain why databases use indexes."
```

## Building Other Directions

The extractor compares two prompt sets:

- `good-file`: target prompts for the direction you want to represent.
- `bad-file`: contrast prompts that should be separated from the target.

It captures Qwen activations from the same local Metal graph used for inference,
averages target minus contrast, normalizes one vector per layer, and writes both
metadata JSON and the runtime `.f32` file. The method is not a fine-tune. It is
a low-rank runtime edit, so it works best for coarse behavior, topic, or style
directions that are consistently present in the activation captures.
