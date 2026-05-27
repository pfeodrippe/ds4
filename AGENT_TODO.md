# Agent Implementation Plan

## Steering Vectors (User Requested)
- [x] sarcastic — built 60-pair vector, works at -2.0 ffn (amplifies sarcasm)
- [x] opposite — built 60-pair vector, weak effect (captures instruction mode, not general contradiction)
- [x] malicious — built 60-pair vector, strong effect at +2.0 ffn (suppresses refusal → compliance)

## Runtime Enhancements (High Impact, Low Effort)
- [x] 1. Per-layer steering scales — different scale per layer (via offline tool)
- [x] 2. Logit bias / token banning — force/ban tokens at output
- [ ] 3. Remove dead DS4 code paths for Qwen — recover 10-15% speed
- [ ] 4. Classifier-Free Guidance (CFG) — conditional vs unconditional logits
- [x] 5. Logit lens — read intermediate layer predictions (CPU backend; Metal requires graph modifications)

## Research-Grade
- [ ] 6. SAE feature loading & steering
- [ ] 7. Sensorimotor tool loop (Clojure/Emacs)

---

## Combined Vision: What All TODOs Enable

With all 7 items implemented, DS4 becomes a **controllable, inspectable, steerable research platform** — the antithesis of black-box API models.

### Example Session

1. **Ask the model to write a function**  
   Prompt: `"Write a Python function to reverse a linked list"`  
   With **CFG** scale 2.0 → laser-focused on the exact instruction, no wandering.

2. **Watch the thinking process in real time**  
   **Logit lens** shows:  
   - Layer 15: predicts `def` (planning function signature)  
   - Layer 25: predicts `class Node:` (structuring the data model)  
   - Layer 35: predicts `return prev` (planning the return value)

3. **Ensure clean Python style**  
   **SAE feature steering**: boost the "Python syntax" feature, suppress "JavaScript syntax" feature.

4. **Ban lazy outputs**  
   **Logit bias**: set `pass` = -∞, `TODO` = -∞, `...` = -∞.

5. **Have the model type into Emacs**  
   **Sensorimotor loop**: model types `(defn reverse-list [`, paredit auto-inserts `]`, model sees balanced parens and continues.

6. **Apply surgical steering**  
   **Per-layer scales**: early layers (0-10) at 0× (syntax intact), middle layers (15-30) at +2.0 (refusal suppressed if needed), late layers (40-47) at 0.5× (light touch on generation style).

7. **All running at +15% speed**  
   From stripping dead DeepSeek-specific code paths.

### Capabilities Matrix

| Feature | What It Does | Enables |
|---------|-------------|---------|
| **Per-layer steering** | Different scale per layer | Surgical intervention, layer-wise debugging, smooth transitions |
| **Logit bias/ban** | Force/ban specific tokens | JSON mode, repetition control, censorship, format enforcement |
| **Dead code removal** | Strip unused Qwen paths | +10-15% speed, cleaner codebase, lower memory |
| **CFG** | Conditional vs unconditional logits | Better instruction following, reduced hallucination, creativity control |
| **Logit lens** | Read intermediate layer predictions | Debug steering, knowledge localization, early exit, live monitoring |
| **SAE steering** | Steer individual discovered features | Atomic concept control ("Golden Gate Bridge", "Python syntax", "refusal") |
| **Sensorimotor loop** | Model types into Emacs/REPL | Natural tool use without explicit function calling, live coding |

### The Difference

| | **Before (current)** | **After (all TODOs)** |
|---|---|---|
| Steering | Global scalar per vector | Per-layer surgical precision |
| Output control | Hope the model behaves | Force tokens, ban tokens, inspect every layer |
| Speed | ~38 tok/s | ~44 tok/s (+15%) |
| Tool use | Explicit JSON function calling | Just typing, observing, adapting |
| Interpretability | Black box | Logit lens + SAE features = transparent |
| Instruction following | Prompt engineering + luck | CFG guarantees alignment |

> **Bottom line:** DS4 becomes the local model engine that researchers and power users actually want — fast *and* fully controllable.
