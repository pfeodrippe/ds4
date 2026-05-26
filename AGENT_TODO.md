# Agent Implementation Plan

## Steering Vectors (User Requested)
- [x] sarcastic — built 60-pair vector, works at -2.0 ffn (amplifies sarcasm)
- [x] opposite — built 60-pair vector, weak effect (captures instruction mode, not general contradiction)
- [x] malicious — built 60-pair vector, strong effect at +2.0 ffn (suppresses refusal → compliance)

## Runtime Enhancements (High Impact, Low Effort)
- [ ] 1. Per-layer steering scales — different scale per layer
- [ ] 2. Logit bias / token banning — force/ban tokens at output
- [ ] 3. Remove dead DS4 code paths for Qwen — recover 10-15% speed
- [ ] 4. Classifier-Free Guidance (CFG) — conditional vs unconditional logits
- [ ] 5. Logit lens — read intermediate layer predictions

## Research-Grade
- [ ] 6. SAE feature loading & steering
- [ ] 7. Sensorimotor tool loop (Clojure/Emacs)
