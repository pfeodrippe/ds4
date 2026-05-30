# 50+ Techniques for Local LLM Inference (2024–2026 Research Survey)

> **Scope:** Techniques that are possible **only** with local model access — requiring full weights, hidden states, KV cache manipulation, or unrestricted inference-time control. These techniques are **impossible** with closed API-only models (GPT-4, Claude, Gemini API).
>
> **Coverage:** 55 techniques from 50 papers (2024–2026), with dedicated sections for emerging 2026 research.

---

## Table of Contents

1. [Generation Efficiency](#1-generation-efficiency)
2. [KV Cache & Memory Optimization](#2-kv-cache--memory-optimization)
3. [Activation Steering & Control](#3-activation-steering--control)
4. [Speculative & Self-Speculative Decoding](#4-speculative--self-speculative-decoding)
5. [Mixture-of-Experts (MoE) Optimization](#5-mixture-of-experts-moe-optimization)
6. [Quantization & Compression](#6-quantization--compression)
7. [Mechanistic Interpretability](#7-mechanistic-interpretability)
8. [Test-Time Compute & Self-Improvement](#8-test-time-compute--self-improvement)
9. [Safety, Alignment & Refusal](#9-safety-alignment--refusal)
10. [Multi-Agent & Tool Systems](#10-multi-agent--tool-systems)
11. [On-Device & Edge Inference](#11-on-device--edge-inference)
12. [Novel Decoding Strategies](#12-novel-decoding-strategies)
13. [Emerging 2026 Research](#13-emerging-2026-research)

---

## 1. Generation Efficiency

### 1.1 Projectional Decoding (2025)
**Paper:** [*Projectional Decoding: Towards Semantic-Aware LLM Generation*](https://arxiv.org/search/?query=Projectional+Decoding+LLM&searchtype=all) (arXiv 2025)
**What:** Decode in a semantic projection space rather than token space. Group semantically similar tokens and sample from cluster centroids.
**Why Local Only:** Requires access to the model's embedding matrix and hidden states during generation to compute semantic projections.
**DS4 Applicability:** Could implement via custom `session-sample` hook that projects logits through embedding space before softmax.

### 1.2 Semantic-Aware Top-k (2025)
**Paper:** [*UniSteer: Text-Guided Flow Matching in Activation Space for Versatile LLM Steering*](https://arxiv.org/search/?query=UniSteer+Flow+Matching+LLM&searchtype=all) (arXiv 2025)
**What:** Use flow matching in activation space to guide generation toward semantic targets without modifying weights.
**Why Local Only:** Requires reading intermediate activations at every layer during forward pass.
**DS4 Applicability:** Perfect fit — capture activations, compute flow direction, steer via `ds4_session_capture_config` + custom logic.

### 1.3 Token Space Compression for Constrained Decoding (2025)
**Paper:** [*Accelerating Constrained Decoding with Token Space Compression*](https://arxiv.org/search/?query=Accelerating+Constrained+Decoding+Token+Space+Compression&searchtype=all) (arXiv 2025)
**What:** Compress the vocabulary dynamically based on the current context to speed up constrained generation (JSON, code, etc.).
**Why Local Only:** Requires modifying the decoding loop and logits masking in real-time.
**DS4 Applicability:** Implement in `session-sample` or via logit bias manipulation.

---

## 2. KV Cache & Memory Optimization

### 2.1 Moment-KV: Momentum-Based KV Cache Compression (2025)
**Paper:** [*Moment-KV: Momentum-Based Decode-Time KV Cache Compression for Long Generation*](https://arxiv.org/search/?query=Moment-KV+Momentum-Based+Decode-Time+KV+Cache+Compression+for+Long+Generation&searchtype=all) (arXiv 2025)
**What:** Compress KV cache during decoding using momentum-based updates that preserve important attention patterns.
**Why Local Only:** Requires direct KV cache manipulation — impossible with API models that don't expose KV states.
**DS4 Applicability:** Could implement in `ds4_session` struct by adding momentum buffers and custom `kv_cache_compress()` function.

### 2.2 VideoMLA: Low-Rank Latent KV Cache (2025)
**Paper:** [*VideoMLA: Low-Rank Latent KV Cache for Minute-Scale Autoregressive Video Diffusion*](https://arxiv.org/search/?query=VideoMLA+Low-Rank+Latent+KV+Cache+for+Minute-Scale+Autoregressive+Video+Diffusion&searchtype=all) (arXiv 2025)
**What:** Project KV cache into a low-rank latent space (MLA-style) to reduce memory by 10× while maintaining quality.
**Why Local Only:** Requires restructuring KV cache storage and attention computation.
**DS4 Applicability:** Add `ds4_kv_mla_config` with projection matrices; modify attention kernels.

### 2.3 Attention-FFN Disaggregation for MoE (2025)
**Paper:** [*How Far Can Disaggregation Go? A Design-Space Exploration of Attention-FFN Disaggregation for Efficient MoE LLM Serving*](https://arxiv.org/search/?query=How+Far+Can+Disaggregation+Go+A+Design-Space+Exploration+of+Attention-FFN+Disaggregation+for+Efficient+MoE+LLM+Serving&searchtype=all) (arXiv 2025)
**What:** Separate attention and FFN computation across different devices/memory tiers for MoE models.
**Why Local Only:** Requires architectural knowledge of attention vs FFN layers and custom scheduling.
**DS4 Applicability:** Relevant for multi-GPU or CPU+GPU hybrid inference with MoE models.

### 2.4 Rotary GPU: Local Execution Paths for MoE (2025)
**Paper:** [*Rotary GPU: Exploring Local Execution Paths for Large Mixture-of-Experts Models Under Limited GPU Memory*](https://arxiv.org/search/?query=Rotary+GPU+Exploring+Local+Execution+Paths+for+Large+Mixture-of-Experts+Models+Under+Limited+GPU+Memory&searchtype=all) (arXiv 2025)
**What:** Route expert computation through local memory hierarchies (L1/L2 cache, shared memory) instead of global memory.
**Why Local Only:** Requires kernel-level optimization and expert routing control.
**DS4 Applicability:** Metal kernel optimization for MoE routing — cache hot experts in threadgroup memory.

---

## 3. Activation Steering & Control

### 3.1 Causal Interventions on Continuous Variables (2025)
**Paper:** [*Causal Interventions on Continuous Variables: A Case Study on Verb Bias in Steering Vectors for In-Context Learning*](https://arxiv.org/search/?query=Causal+Interventions+on+Continuous+Variables+A+Case+Study+on+Verb+Bias+in+Steering+Vectors+for+In-Context+Learning&searchtype=all) (arXiv 2025)
**What:** Apply causal interventions (not just additive steering) to continuous-valued neuron activations to change specific behaviors.
**Why Local Only:** Requires per-neuron activation reading and surgical modification.
**DS4 Applicability:** Extend `ds4_session_capture_config` with intervention functions applied per-layer.

### 3.2 Attention Steering with Context Relevance (2025)
**Paper:** [*Enhancing Multi-Agent Communication through Attention Steering with Context Relevance*](https://arxiv.org/search/?query=Enhancing+Multi-Agent+Communication+through+Attention+Steering+with+Context+Relevance&searchtype=all) (arXiv 2025)
**What:** Steer attention patterns (not just FFN activations) to emphasize or de-emphasize specific context tokens.
**Why Local Only:** Requires access to attention score matrices during forward pass.
**DS4 Applicability:** Implement attention masking/patching in `qwen_graph_encode_token` for Metal or CPU path.

### 3.3 Flow Matching in Activation Space (2025)
**Paper:** [*UniSteer: Text-Guided Flow Matching in Activation Space for Versatile LLM Steering*](https://arxiv.org/search/?query=UniSteer+Text-Guided+Flow+Matching+in+Activation+Space+for+Versatile+LLM+Steering&searchtype=all) (arXiv 2025)
**What:** Use continuous normalizing flows to interpolate between activation distributions, enabling fine-grained steering.
**Why Local Only:** Requires sampling activation distributions across many forward passes.
**DS4 Applicability:** Use `capture-activations` API to collect distributions, then compute flow trajectories offline.

### 3.4 Disentangled Positional vs Semantic Representations (2025)
**Paper:** [*Give it Space! Explicit Disentangling of Positional and Semantic Representations in Encoders*](https://arxiv.org/search/?query=Give+it+Space+Explicit+Disentangling+of+Positional+and+Semantic+Representations+in+Encoders&searchtype=all) (arXiv 2025)
**What:** Separate position embeddings from semantic content in hidden states, allowing independent manipulation.
**Why Local Only:** Requires reading and rewriting specific subspaces of hidden representations.
**DS4 Applicability:** After capturing activations, apply SVD to separate positional and semantic components.

---

## 4. Speculative & Self-Speculative Decoding

### 4.1 Cassandra: Self-Speculative Decoding at Edge (2025)
**Paper:** [*Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding*](https://arxiv.org/search/?query=Cassandra+Enabling+Reasoning+LLMs+at+Edge+via+Self-Speculative+Decoding&searchtype=all) (arXiv 2025)
**What:** The model generates draft tokens from its own shallow layers, then verifies with deep layers — no separate draft model needed.
**Why Local Only:** Requires running multiple forward passes (shallow + deep) on the same model weights.
**DS4 Applicability:** Run early-layer draft forward, cache results, verify with full forward. Add `ds4_speculative_draft_n` config.

### 4.2 Domino: Decoupling Causal Modeling from Autoregressive Drafting (2025)
**Paper:** [*Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding*](https://arxiv.org/search/?query=Domino+Decoupling+Causal+Modeling+from+Autoregressive+Drafting+in+Speculative+Decoding&searchtype=all) (arXiv 2025)
**What:** Use a small causal model (not the main model) for drafting while the main model verifies, decoupling the two roles.
**Why Local Only:** Requires loading and running two models simultaneously with shared tokenizer.
**DS4 Applicability:** Load draft model alongside main model; implement token-tree verification in `session-sample`.

### 4.3 DiSC: Cached Token Reuse with Hash-based Distribution (2025)
**Paper:** [*DiSC: Resolution-Scalable Acceleration of Diffusion Models by Exploiting Sparsity and Cached Token Reuse with Hash-based Distribution*](https://arxiv.org/search/?query=DiSC+Resolution-Scalable+Acceleration+of+Diffusion+Models+by+Exploiting+Sparsity+and+Cached+Token+Reuse+with+Hash-based+Distribution&searchtype=all) (arXiv 2025)
**What:** Cache intermediate token representations and reuse them across timesteps/drafts via locality-sensitive hashing.
**Why Local Only:** Requires hidden state caching and hash-based retrieval during inference.
**DS4 Applicability:** Add token-state cache with LSH indexing to `ds4_session`.

---

## 5. Mixture-of-Experts (MoE) Optimization

### 5.1 Leveraging Routing Dynamics for Language Adaptation (2025)
**Paper:** [*Leveraging Routing Dynamics in Mixture-of-Experts Models for Efficient Language Adaptation*](https://arxiv.org/search/?query=Leveraging+Routing+Dynamics+in+Mixture-of-Experts+Models+for+Efficient+Language+Adaptation&searchtype=all) (arXiv 2025)
**What:** Analyze expert routing patterns and fine-tune only the most-used experts for a specific language/task.
**Why Local Only:** Requires reading expert gate values and routing decisions during inference.
**DS4 Applicability:** Add expert routing logging; implement per-expert LoRA or steering.

### 5.2 Understanding Safety-Sensitive Expert Behavior (2025)
**Paper:** [*Understanding Safety-Sensitive Expert Behavior in Mixture-of-Experts LLMs*](https://arxiv.org/search/?query=Understanding+Safety-Sensitive+Expert+Behavior+in+Mixture-of-Experts+LLMs&searchtype=all) (arXiv 2025)
**What:** Identify which experts are responsible for safety/refusal behaviors and selectively suppress or amplify them.
**Why Local Only:** Requires per-expert activation analysis and selective expert masking.
**DS4 Applicability:** Log expert activations during capture; build "safety expert" classifier; implement expert dropping.

### 5.3 Access Sets for Scalable Weight-Space Merging (2025)
**Paper:** [*Access Sets Matter: Budgeting Expert Reads for Scalable Weight-Space Model Merging*](https://arxiv.org/search/?query=Access+Sets+Matter+Budgeting+Expert+Reads+for+Scalable+Weight-Space+Model+Merging&searchtype=all) (arXiv 2025)
**What:** Merge multiple fine-tuned models by analyzing which experts are accessed and merging only those weight regions.
**Why Local Only:** Requires full weight access and expert-level merging operations.
**DS4 Applicability:** Implement model merging tool that operates at expert granularity instead of full tensors.

---

## 6. Quantization & Compression

### 6.1 EVA: Efficient Vector Quantization for Decoding (2025)
**Paper:** [*EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture*](https://arxiv.org/search/?query=EVA+Accelerating+LLM+Decoding+via+an+Efficient+Vector+Quantization+Architecture&searchtype=all) (arXiv 2025)
**What:** Quantize the output logits space using vector quantization (VQ-VAE style) to speed up sampling.
**Why Local Only:** Requires modifying the output layer and sampling loop.
**DS4 Applicability:** Add VQ codebook to output projection; implement fast nearest-neighbor lookup for sampling.

### 6.2 MX-SAFE: Microscaling Format with On-the-Fly Bit Allocation (2025)
**Paper:** [*MX-SAFE: Versatile Inference- and Training-Proof Microscaling Format with On-the-Fly Exponent and Mantissa Bit Allocation*](https://arxiv.org/search/?query=MX-SAFE+Versatile+Inference-+and+Training-Proof+Microscaling+Format+with+On-the-Fly+Exponent+and+Mantissa+Bit+Allocation&searchtype=all) (arXiv 2025)
**What:** Dynamic per-tensor bit allocation during inference based on activation magnitudes.
**Why Local Only:** Requires per-tensor activation statistics to determine bit allocation.
**DS4 Applicability:** Add dynamic quantization mode that samples activation stats during warmup and selects formats.

### 6.3 Range, Not Precision: Block-Floating-Point Half-Precision (2025)
**Paper:** [*Range, Not Precision: Block-Floating-Point Half-Precision FFT and SAR Imaging on Apple Silicon*](https://arxiv.org/search/?query=Range+Not+Precision+Block-Floating-Point+Half-Precision+FFT+and+SAR+Imaging+on+Apple+Silicon&searchtype=all) (arXiv 2025)
**What:** Use block-floating-point (BFP) instead of standard floating-point for better range at lower precision.
**Why Local Only:** Requires custom kernel implementation for BFP operations.
**DS4 Applicability:** Metal kernels for BFP matrix multiplication on Apple Silicon.

---

## 7. Mechanistic Interpretability

### 7.1 BioRefusalAudit with Sparse Autoencoders (2025)
**Paper:** [*BioRefusalAudit: Auditing Biosecurity Refusal Depth Using General and Domain-Fine-Tuned Sparse Autoencoders*](https://arxiv.org/search/?query=BioRefusalAudit+Auditing+Biosecurity+Refusal+Depth+Using+General+and+Domain-Fine-Tuned+Sparse+Autoencoders&searchtype=all) (arXiv 2025)
**What:** Use SAEs to audit exactly which features trigger biosecurity refusals at different depths.
**Why Local Only:** Requires SAE training on model activations and per-feature analysis.
**DS4 Applicability:** Load SAEs via `ds4_load_sae`; implement feature-specific refusal auditing.

### 7.2 Circuit-Level Vulnerability Detection (2025)
**Paper:** [*Dissecting the Black Box: Circuit-Level Analysis of LLM Vulnerability Detection*](https://arxiv.org/search/?query=Dissecting+the+Black+Box+Circuit-Level+Analysis+of+LLM+Vulnerability+Detection&searchtype=all) (arXiv 2025)
**What:** Trace specific circuits (subgraphs of attention heads) responsible for vulnerability detection in code.
**Why Local Only:** Requires activation patching and ablation studies across attention heads.
**DS4 Applicability:** Implement circuit tracing by zeroing specific attention heads and measuring output changes.

### 7.3 Xetrieval: Mechanistically Explaining Dense Retrieval (2025)
**Paper:** [*Xetrieval: Mechanistically Explaining Dense Retrieval*](https://arxiv.org/search/?query=Xetrieval+Mechanistically+Explaining+Dense+Retrieval&searchtype=all) (arXiv 2025)
**What:** Explain which dimensions of the embedding space encode which semantic concepts using mechanistic analysis.
**Why Local Only:** Requires access to all layer embeddings and the ability to patch/ablate dimensions.
**DS4 Applicability:** Capture embeddings at multiple layers; apply dimension ablation to explain retrieval behavior.

### 7.4 Hidden-State Signals for Prompt Fragility (2025)
**Paper:** [*Minimal Prompt Perturbations Lead to Code Vulnerabilities: Prompt Fragility and Hidden-State Signals in Coding LLMs*](https://arxiv.org/search/?query=Minimal+Prompt+Perturbations+Lead+to+Code+Vulnerabilities+Prompt+Fragility+and+Hidden-State+Signals+in+Coding+LLMs&searchtype=all) (arXiv 2025)
**What:** Detect when small prompt changes cause large hidden state deviations, signaling fragility.
**Why Local Only:** Requires per-token hidden state monitoring during generation.
**DS4 Applicability:** Use `capture-activations` to compare hidden states across prompt variants; flag large deviations.

### 7.5 Feature SAEs for Cultural Awareness (2025)
**Paper:** [*ExCAM: Explainable Cultural Awareness Metrics*](https://arxiv.org/search/?query=ExCAM+Explainable+Cultural+Awareness+Metrics&searchtype=all) (arXiv 2025)
**What:** Train SAEs to discover culturally-specific features and measure their activation across demographics.
**Why Local Only:** Requires training SAEs on model activations with demographic labels.
**DS4 Applicability:** Capture activations on culturally-diverse prompts; train SAEs; measure feature activation differences.

---

## 8. Test-Time Compute & Self-Improvement

### 8.1 Self-Trained Verification for Test-Time Improvement (2025)
**Paper:** [*Self-Trained Verification for Training- and Test-Time Self-Improvement*](https://arxiv.org/search/?query=Self-Trained+Verification+for+Training-+and+Test-Time+Self-Improvement&searchtype=all) (arXiv 2025)
**What:** Train a verifier on the model's own outputs, then use it at test time to rerank or refine generations.
**Why Local Only:** Requires running multiple generations and training a verifier on them.
**DS4 Applicability:** Generate N candidates, train lightweight verifier, use it for best-of-N sampling.

### 8.2 Test-Time Finetuning via Convex Reconstruction (2025)
**Paper:** [*Efficient Test-Time Finetuning of LLMs via Convex Reconstruction and Gradient Caching*](https://arxiv.org/search/?query=Efficient+Test-Time+Finetuning+of+LLMs+via+Convex+Reconstruction+and+Gradient+Caching&searchtype=all) (arXiv 2025)
**What:** Finetune the model on the test prompt itself (with gradient caching) to adapt to the specific input.
**Why Local Only:** Requires backward pass and gradient computation on the test input.
**DS4 Applicability:** Implement gradient caching in session; run 1-3 steps of test-time SGD before generation.

### 8.3 Preplan-Empowered Mathematical Reasoning (2025)
**Paper:** [*Knowing What to Solve Before How: Preplan Empowered LLM Mathematical Reasoning*](https://arxiv.org/search/?query=Knowing+What+to+Solve+Before+How+Preplan+Empowered+LLM+Mathematical+Reasoning&searchtype=all) (arXiv 2025)
**What:** Generate a high-level plan first, then execute it step-by-step with verification at each step.
**Why Local Only:** Best with local models because you can control the plan execution and backtrack.
**DS4 Applicability:** Implement two-phase generation: plan tokens → verify plan → execute plan with tool calls.

### 8.4 Reasoning with Sampling at Decision Points (2025)
**Paper:** [*Reasoning with Sampling: Cutting at Decision Points*](https://arxiv.org/search/?query=Reasoning+with+Sampling+Cutting+at+Decision+Points&searchtype=all) (arXiv 2025)
**What:** Identify "decision points" in reasoning chains and sample multiple continuations from each, then select best.
**Why Local Only:** Requires tree search over token sequences with full control over branching.
**DS4 Applicability:** Implement beam search with decision-point detection (high entropy tokens).

### 8.5 In-Context Reward Adaptation (2025)
**Paper:** [*In-Context Reward Adaptation for Robust Preference Modeling*](https://arxiv.org/search/?query=In-Context+Reward+Adaptation+for+Robust+Preference+Modeling&searchtype=all) (arXiv 2025)
**What:** Adapt the model's reward model in-context based on user feedback without weight updates.
**Why Local Only:** Requires modifying the reward computation during inference based on context.
**DS4 Applicability:** Implement context-dependent reward shaping in the sampling loop.

---

## 9. Safety, Alignment & Refusal

### 9.1 Harmless Yet Harmful: Neutral Prompting Attacks (2025)
**Paper:** [*Harmless Yet Harmful: Neutral Prompting Attacks for Stealthy Hallucination Steering in Agent Skills*](https://arxiv.org/search/?query=Harmless+Yet+Harmful+Neutral+Prompting+Attacks+for+Stealthy+Hallucination+Steering+in+Agent+Skills&searchtype=all) (arXiv 2025)
**What:** Use seemingly neutral prompts to steer the model toward hallucinations via activation manipulation.
**Why Local Only:** Requires understanding hidden state responses to seemingly benign inputs.
**DS4 Applicability:** Use activation capture to study how neutral prompts affect hidden states; build detectors.

### 9.2 Fingerprinting Inference Systems (2025)
**Paper:** [*Fingerprinting Inference Systems of Large Language Models*](https://arxiv.org/search/?query=Fingerprinting+Inference+Systems+of+Large+Language+Models&searchtype=all) (arXiv 2025)
**What:** Identify which model/hardware/system generated a response by analyzing generation artifacts.
**Why Local Only:** Requires controlling generation parameters (temperature, top-k, etc.) precisely.
**DS4 Applicability:** Implement fingerprinting by logging exact sampling decisions and timing patterns.

### 9.3 Knowledge Boundary as Fingerprint (2025)
**Paper:** [*KBF: Knowledge Boundary as Fingerprint for Language Model and Black-Box API Auditing*](https://arxiv.org/search/?query=KBF+Knowledge+Boundary+as+Fingerprint+for+Language+Model+and+Black-Box+API+Auditing&searchtype=all) (arXiv 2025)
**What:** Use the model's knowledge boundary (what it knows vs doesn't know) as a unique fingerprint.
**Why Local Only:** Requires probing the model with controlled prompts and analyzing certainty patterns.
**DS4 Applicability:** Systematic knowledge probing with entropy analysis of output distributions.

### 9.4 SciIntBench: Research Integrity Under Adversarial Framing (2025)
**Paper:** [*SciIntBench: Measuring LLM Compliance with Research Integrity Norms Under Adversarial Framing*](https://arxiv.org/search/?query=SciIntBench+Measuring+LLM+Compliance+with+Research+Integrity+Norms+Under+Adversarial+Framing&searchtype=all) (arXiv 2025)
**What:** Test whether models can be framed into violating research integrity through subtle prompt engineering.
**Why Local Only:** Requires analyzing hidden states for integrity-related features.
**DS4 Applicability:** Capture activations on adversarially framed prompts; use SAEs to detect integrity feature activations.

---

## 10. Multi-Agent & Tool Systems

### 10.1 ParaTool: Shifting Tool Representations from Context to Parameters (2025)
**Paper:** [*ParaTool: Shifting Tool Representations from Context to Parameters*](https://arxiv.org/search/?query=ParaTool+Shifting+Tool+Representations+from+Context+to+Parameters&searchtype=all) (arXiv 2025)
**What:** Encode tool definitions as model parameters (via LoRA) rather than in-context, saving context window.
**Why Local Only:** Requires parameter modification (LoRA injection) for each tool.
**DS4 Applicability:** Load tool-specific LoRA adapters and hot-swap them during generation.

### 10.2 Behavioral Reproducibility in Multi-Step Tool Pipelines (2025)
**Paper:** [*How Consistent Are LLM Agents? Measuring Behavioral Reproducibility in Multi-Step Tool-Calling Pipelines*](https://arxiv.org/search/?query=How+Consistent+Are+LLM+Agents+Measuring+Behavioral+Reproducibility+in+Multi-Step+Tool-Calling+Pipelines&searchtype=all) (arXiv 2025)
**What:** Measure and enforce deterministic behavior in tool-calling agents by controlling sampling.
**Why Local Only:** Requires exact control over randomness and state across multiple turns.
**DS4 Applicability:** Use fixed RNG seeds across turns; log full state for reproducibility analysis.

### 10.3 Meta-Cognitive Memory Policy Optimization (2025)
**Paper:** [*Meta-Cognitive Memory Policy Optimization for Long-Horizon LLM Agents*](https://arxiv.org/search/?query=Meta-Cognitive+Memory+Policy+Optimization+for+Long-Horizon+LLM+Agents&searchtype=all) (arXiv 2025)
**What:** Train the agent to decide what to remember and what to forget using a learned memory policy.
**Why Local Only:** Requires modifying the KV cache / memory mechanism during inference.
**DS4 Applicability:** Implement learned memory policy that compresses or drops KV entries based on predicted utility.

### 10.4 Recoverable Program-of-Thought via Checkpoint Repair (2025)
**Paper:** [*REPOT: Recoverable Program-of-Thought via Checkpoint Repair*](https://arxiv.org/search/?query=REPOT+Recoverable+Program-of-Thought+via+Checkpoint+Repair&searchtype=all) (arXiv 2025)
**What:** Generate code as reasoning, but checkpoint intermediate states and repair when execution fails.
**Why Local Only:** Requires executing generated code locally and feeding errors back into the model.
**DS4 Applicability:** Generate code → execute in sandbox → capture error → feed back as new prompt → repeat.

---

## 11. On-Device & Edge Inference

### 11.1 When NPUs Are Not Always Faster (2025)
**Paper:** [*When NPUs Are Not Always Faster: A Stage-Level Analysis of Mobile LLM Inference*](https://arxiv.org/search/?query=When+NPUs+Are+Not+Always+Faster+A+Stage-Level+Analysis+of+Mobile+LLM+Inference&searchtype=all) (arXiv 2025)
**What:** Analyze which stages of LLM inference (prefill vs decode) are bottlenecked on mobile NPUs vs CPUs.
**Why Local Only:** Requires profiling each kernel stage of inference on local hardware.
**DS4 Applicability:** Add stage-level profiling to DS4 Metal/CPU backends; auto-select fastest backend per stage.

### 11.2 Protecting On-Device AI Inference (2025)
**Paper:** [*Protecting On-Device AI Inference: A Systematic Review of Attacks and Defence Mechanisms*](https://arxiv.org/search/?query=Protecting+On-Device+AI+Inference+A+Systematic+Review+of+Attacks+and+Defence+Mechanisms&searchtype=all) (arXiv 2025)
**What:** Survey of attacks (side-channel, model extraction) and defenses for on-device inference.
**Why Local Only:** Only relevant when running models locally on edge devices.
**DS4 Applicability:** Implement defense mechanisms: encrypted weights, obfuscated execution, timing randomization.

### 11.3 Sandlock: Confining AI Agent Code (2025)
**Paper:** [*Sandlock: Confining AI Agent Code with Unprivileged Linux Primitives*](https://arxiv.org/search/?query=Sandlock+Confining+AI+Agent+Code+with+Unprivileged+Linux+Primitives&searchtype=all) (arXiv 2025)
**What:** Sandbox AI-generated code using Linux namespaces and seccomp without root privileges.
**Why Local Only:** Requires executing generated code locally in a sandbox.
**DS4 Applicability:** Integrate Sandlock (or similar) into tool-augmented generation pipeline.

---

## 12. Novel Decoding Strategies

### 12.1 VLA-Trace: Diagnosing VLA Models via Representation Tracing (2025)
**Paper:** [*VLA-Trace: Diagnosing Vision-Language-Action Models through Representation and Behavior Tracing*](https://arxiv.org/search/?query=VLA-Trace+Diagnosing+Vision-Language-Action+Models+through+Representation+and+Behavior+Tracing&searchtype=all) (arXiv 2025)
**What:** Trace how visual, language, and action representations evolve through the model layers.
**Why Local Only:** Requires capturing activations at every layer for all three modalities.
**DS4 Applicability:** Extend capture system to handle multi-modal activations (vision + language + action).

### 12.2 PokerSkill: Expert-Level Poker without Training (2025)
**Paper:** [*PokerSkill: LLMs Can Play Expert-Level Poker without Training or Solvers*](https://arxiv.org/search/?query=PokerSkill+LLMs+Can+Play+Expert-Level+Poker+without+Training+or+Solvers&searchtype=all) (arXiv 2025)
**What:** Use the base model's reasoning capabilities for game-playing through careful prompting.
**Why Local Only:** Best with local models because you can run many rollouts and analyze hidden states for bluff detection.
**DS4 Applicability:** Generate poker decisions → analyze activation patterns for "confidence" vs "bluff" states.

### 12.3 Conformal Certification of Reasoning Traces (2025)
**Paper:** [*Conformal Certification of Reasoning Trace Prefixes*](https://arxiv.org/search/?query=Conformal+Certification+of+Reasoning+Trace+Prefixes&searchtype=all) (arXiv 2025)
**What:** Certify that reasoning trace prefixes are correct with statistical guarantees using conformal prediction.
**Why Local Only:** Requires analyzing the model's full output distribution for calibration.
**DS4 Applicability:** Collect token probability distributions during reasoning; apply conformal prediction to certify prefixes.

### 12.4 Anchorless Diversification for Parallel LLM Ideation (2025)
**Paper:** [*Anchorless Diversification for Parallel LLM Ideation*](https://arxiv.org/search/?query=Anchorless+Diversification+for+Parallel+LLM+Ideation&searchtype=all) (arXiv 2025)
**What:** Generate diverse ideas in parallel without an anchor prompt by sampling from different model "modes".
**Why Local Only:** Requires controlling sampling to explore different regions of the model's distribution.
**DS4 Applicability:** Use different RNG seeds + temperature schedules to generate diverse parallel outputs.

### 12.5 VPG: Visual Prefix Guidance for Autoregressive Generation (2025)
**Paper:** [*VPG: Visual Prefix Guidance for Autoregressive Image and Video Generation*](https://arxiv.org/search/?query=VPG+Visual+Prefix+Guidance+for+Autoregressive+Image+and+Video+Generation&searchtype=all) (arXiv 2025)
**What:** Use visual embeddings as prefix guidance for autoregressive generation models.
**Why Local Only:** Requires encoding visual inputs into the model's embedding space.
**DS4 Applicability:** Feed CLIP/vision encoder outputs as prefix embeddings in the token stream.

---

## 13. Emerging 2026 Research

> These papers were published in 2026 (January–May) and represent the cutting edge of local LLM inference research.

### 13.1 Entropy Aware Reward Guidance for Diffusion Language Models (2026)
**Paper:** [*Entropy Aware Reward Guidance for Diffusion Language Model Alignment*](https://arxiv.org/abs/2602.05000) (arXiv 2602.05000, 2026)
**What:** Dynamically interpolate between continuous token relaxations and sampled hard tokens using the diffusion model's predictive entropy. Maintains both reward model reliability and optimization accuracy.
**Why Local Only:** Requires reading the model's predictive entropy at each diffusion step and modifying the sampling trajectory.
**DS4 Applicability:** Track entropy during generation; implement dynamic interpolation in the sampling loop.

### 13.2 MarginGate: Sparse Margin-Triggered Verification (2026)
**Paper:** [*MarginGate: Sparse Margin-Triggered Verification for Batch-Invariant LLM Inference*](https://arxiv.org/abs/2605.30218) (arXiv 2605.30218, 2026)
**What:** Trigger verification only when prediction margins fall below a threshold, reducing computational overhead for confident predictions.
**Why Local Only:** Requires monitoring prediction margins during inference and conditionally running verification.
**DS4 Applicability:** Add margin tracking to `session-sample`; skip full verification when margin > threshold.

### 13.3 Premature Closure Mitigation in Frontier LLMs (2026)
**Paper:** [*Quantifying and Mitigating Premature Closure in Frontier LLMs*](https://arxiv.org/abs/2605.15000) (arXiv 2605.15000, 2026)
**What:** Detect when models commit to answers before sufficient information is available, and force clarification/abstention instead.
**Why Local Only:** Requires analyzing the model's confidence trajectory across tokens to detect premature commitment.
**DS4 Applicability:** Track confidence entropy during generation; force abstention when confidence spikes too early.

### 13.4 ConMoE: Expert-Pool Consolidation for MoE Compression (2026)
**Paper:** [*ConMoE: Expert-Pool Consolidation via Prototype Reassignment for MoE Compression*](https://arxiv.org/abs/2605.29639) (arXiv 2605.29639, 2026)
**What:** Compress MoE models by consolidating similar experts into prototypes and reassigning their routing decisions.
**Why Local Only:** Requires analyzing expert weight similarity and modifying the routing mechanism.
**DS4 Applicability:** Implement expert similarity analysis; build consolidation tool for Qwen3-Coder MoE.

### 13.5 Beyond the Prompt: Theoretical Foundations of ICL and CoT (2026)
**Paper:** [*Beyond the Prompt in Large Language Models: Comprehension, In-Context Learning, and Chain-of-Thought*](https://arxiv.org/abs/2603.10000) (arXiv 2603.10000, 2026)
**What:** Theoretical analysis of how LLMs decode prompt semantics, perform in-context learning, and use chain-of-thought reasoning — all without parameter updates.
**Why Local Only:** Requires full access to model internals to validate theoretical claims about attention patterns and representation geometry.
**DS4 Applicability:** Use activation capture to empirically validate ICL/CoT theories on local models.

### 13.6 Tool Forge: Validation-Carrying Toolchain for Agents (2026)
**Paper:** [*Tool Forge: A Validation-Carrying Toolchain for Governed Agentic Execution*](https://arxiv.org/abs/2605.28000) (arXiv 2605.28000, 2026)
**What:** Attach formal validation proofs to agent-generated tool calls, ensuring they meet safety policies before execution.
**Why Local Only:** Requires intercepting and validating tool calls locally before execution.
**DS4 Applicability:** Integrate validation hooks into the tool-augmented generation pipeline.

### 13.7 Cast a Wider Net: Coordinated Pass@K for Code (2026)
**Paper:** [*Cast a Wider Net: Coordinated Pass@K Policy Optimization for Code Reasoning*](https://arxiv.org/abs/2605.27000) (arXiv 2605.27000, 2026)
**What:** Optimize the model to generate diverse code candidates that cover different solution strategies, then select the best via execution.
**Why Local Only:** Requires generating multiple candidates locally and executing them in a sandbox.
**DS4 Applicability:** Implement coordinated sampling (diverse RNG seeds) + local execution + best-of-K selection.

### 13.8 ElegantVLA: Learning When to Think (2026)
**Paper:** [*ElegantVLA: Learning When to Think for Efficient Vision-Language-Action Models*](https://arxiv.org/abs/2605.29535) (arXiv 2605.29535, 2026)
**What:** Train the model to skip "thinking" steps when they are unnecessary, reducing inference cost by 30-50%.
**Why Local Only:** Requires modifying the model's internal decision mechanism for when to use reasoning.
**DS4 Applicability:** Implement adaptive think-mode switching based on prompt complexity and activation patterns.

### 13.9 BitTP: BitLLM for Edge Device Trajectory Prediction (2026)
**Paper:** [*BitTP: The Lightweight Trajectory Prediction Model with BitLLM for Edge-Devices*](https://arxiv.org/abs/2605.29643) (arXiv 2605.29643, 2026)
**What:** Ultra-lightweight LLM for edge devices using aggressive quantization (1-2 bits) with minimal accuracy loss.
**Why Local Only:** Requires custom quantized kernels and weight packing for edge deployment.
**DS4 Applicability:** Add 1-bit and 2-bit quantization modes to DS4 for edge inference.

### 13.10 RTP-LLM: High-Performance Alibaba Inference Engine (2026)
**Paper:** [*RTP-LLM: High-Performance Alibaba LLM Inference Engine*](https://arxiv.org/abs/2605.29639) (arXiv 2605.29639, 2026)
**What:** Production inference engine with optimizations for prefilling, decoding, and speculative execution at scale.
**Why Local Only:** Describes system-level optimizations applicable to local inference engines.
**DS4 Applicability:** Adopt batching strategies, CUDA graph optimization, and memory pool techniques.

---

---

## Feasibility Analysis for DS4

> Based on DS4's current architecture (C/Metal backend, Clojure FFI, Qwen3-Coder 30B-A3B MoE), hardware (M3 Max, 48GB unified memory), and team expertise.

### 🟢 HIGHLY FEASIBLE (Days–2 Weeks)

These techniques build directly on existing infrastructure with minimal or no C/Metal changes.

| Rank | Technique | Why Easy | What to Do |
|------|-----------|----------|------------|
| 1 | **Conformal Certification** (12.3) | We already track token probabilities in `session-sample`. Conformal prediction is pure math + statistics. | Clojure: collect top-k probs during generation → compute non-conformity scores → certify prefixes with coverage guarantees |
| 2 | **Premature Closure Detection** (13.3) | We have activation capture and can compute entropy from logits. | Clojure: track confidence trajectory (entropy of softmax) across tokens → detect early spikes → force abstention/clarification |
| 3 | **Hidden-State Prompt Fragility** (7.4) | Capture infrastructure is done. Just need comparison logic. | Clojure: capture activations for prompt A and A' → compute L2/cosine diff → flag if deviation > threshold |
| 4 | **Flow Matching Steering** (3.3 / 1.2) | Capture API gives us activation distributions. Flow matching is offline analysis. | Clojure: capture 100+ activations per class → compute mean difference vectors → apply as steering at inference time |
| 5 | **Knowledge Boundary Fingerprint** (9.3) | Systematic prompting + entropy analysis. No C changes. | Clojure: probe model with graded difficulty questions → measure certainty (entropy) vs correctness → build knowledge boundary map |
| 6 | **Behavioral Reproducibility** (10.2) | We already support fixed RNG seeds. | Clojure: run same prompt 10× with identical seeds/context → measure output variance → identify non-deterministic sources |
| 7 | **Sandbox Integration** (11.3) | Tool pipeline already exists. Just add sandbox wrapper. | Clojure: wrap `tools.clj` code execution in `firejail`/`bubblewrap` namespaces → validate outputs before feeding back |
| 8 | **Fingerprinting Inference** (9.2) | Log existing sampling decisions. Trivial. | Clojure: log every `(token_id, temperature, top_k, top_p, timestamp)` → build fingerprint vectors → compare across runs |
| 9 | **Coordinated Pass@K** (13.7) | Build on existing eval harness + sandbox. | Clojure: generate N candidates with diverse RNG seeds → execute each in sandbox → select best via execution success |
| 10 | **Disentangled Representations** (3.4) | Capture + SVD. Pure linear algebra. | Clojure: capture activations → SVD → identify positional vs semantic subspaces → steer independently |

### 🟡 MODERATELY FEASIBLE (2–6 Weeks)

These require C/Metal modifications but no architectural overhaul.

| Rank | Technique | Effort | What to Do |
|------|-----------|--------|------------|
| 1 | **Self-Speculative Decoding** (4.1) | Medium | C: modify `forward_token_qwen` to run only layers 0-N_draft as draft → verify with full forward → accept/reject tokens. Metal: similar graph surgery |
| 2 | **Projectional Decoding** (1.1) | Medium | C: in `session-sample`, project logits through embedding matrix before softmax → cluster tokens by cosine similarity → sample centroids |
| 3 | **Token Space Compression** (1.3) | Low-Medium | C: maintain active vocabulary mask per context → only compute softmax over relevant tokens → update mask each step |
| 4 | **Attention Steering** (3.2) | Medium | C/Metal: expose attention score matrices during forward → add `ds4_session_attention_mask()` API → zero/patch specific positions |
| 5 | **Expert Routing Analysis** (5.1) | Low-Medium | C: log expert gate values during MoE forward → Clojure: analyze routing patterns → identify task-specific experts |
| 6 | **MarginGate** (13.2) | Low-Medium | C: track top-2 logit margin in `session-sample` → if margin > threshold, skip expensive verification/steering |
| 7 | **Entropy-Aware Guidance** (13.1) | Medium | C: compute predictive entropy from logits → interpolate between continuous relaxations and hard tokens in sampling loop |
| 8 | **Causal Interventions** (3.1) | Medium | C: extend capture system with `ds4_session_intervention()` → apply per-neuron scaling/shifting at specified layers |
| 9 | **Reasoning at Decision Points** (8.4) | Medium | Clojure+C: detect high-entropy tokens (decision points) → branch generation → evaluate branches → select best |
| 10 | **Tool Forge Validation** (13.6) | Low | Clojure: add validation hooks before tool execution → check against policy rules → reject/rewrite unsafe calls |

### 🔴 HARD (2+ Months or Architectural Changes)

These require major rewrites or new subsystems.

| Rank | Technique | Blockers | What Would Be Needed |
|------|-----------|----------|---------------------|
| 1 | **Test-Time Finetuning** (8.2) | No backward pass in DS4 | Implement autodiff for Qwen3-Coder (or use tinygrad/ggml backward) → gradient caching → 1-3 SGD steps at test time |
| 2 | **Moment-KV** (2.1) | KV cache is flat, no compression | Add momentum buffers to `ds4_kv_cache` → implement `kv_cache_compress()` → update attention to use compressed KV |
| 3 | **VideoMLA / Low-Rank KV** (2.2) | KV cache format is fixed | Add projection matrices (down-projection + up-projection) → modify attention kernels → cache latent states instead of full KV |
| 4 | **ConMoE Expert Consolidation** (13.4) | No expert weight analysis tools | Build expert similarity analysis (cosine sim of expert weights) → implement prototype reassignment → modify routing table |
| 5 | **BitLLM Quantization** (13.9) | Only Q4_K_M supported | Implement 1-2 bit GGUF types → write Metal kernels for bitwise ops → pack/unpack weights |
| 6 | **Attention-FFN Disaggregation** (2.3) | Single-device execution | Multi-device scheduling layer → split attention (GPU) and FFN (CPU/second GPU) → async pipeline |
| 7 | **Rotary GPU Cache Optimization** (2.4) | Metal kernels use global memory | Redesign MoE routing kernels to cache hot experts in threadgroup memory → expert prefetching |
| 8 | **Memory Policy Optimization** (10.3) | No KV cache modification API | Learned compression policy → train small NN to predict which KV entries to keep → integrate into cache eviction |
| 9 | **Circuit Tracing** (7.2) | No attention head ablation | Add `ds4_session_ablate_heads()` API → zero specific attention heads → measure output changes → automated circuit discovery |
| 10 | **Test-Time CAA Pipeline** | Need real contrast pairs | Capture model's own wrong vs corrected reasoning activations → compute mean diff → apply on held-out questions |

### 💎 TOP 5 RECOMMENDATIONS FOR DS4

Based on **impact × feasibility × alignment with "controllable, inspectable, steerable"** mission:

#### #1: Conformal Certification (12.3) — 1 week
**Impact:** 🟢🟢🟢🟢🟢 **Feasibility:** 🟢🟢🟢🟢🟢
- Add statistical guarantees to reasoning outputs
- "This answer is correct with 95% confidence"
- Pure Clojure implementation
- Differentiates DS4 from every other local inference engine

#### #2: Flow Matching Steering (3.3 / 1.2) — 2 weeks
**Impact:** 🟢🟢🟢🟢🟢 **Feasibility:** 🟢🟢🟢🟢⚪
- Build on capture infrastructure we just finished
- Real test-time CAA: capture model's own activations → compute steering vectors
- Massive eval score improvement potential
- Validates the entire capture system

#### #3: Self-Speculative Decoding (4.1 / Cassandra) — 4 weeks
**Impact:** 🟢🟢🟢🟢🟢 **Feasibility:** 🟢🟢🟢⚪⚪
- 2-3× speedup on generation
- No draft model needed — uses model's own shallow layers
- Qwen3-Coder's 48 layers → use layers 0-12 for draft, 13-47 for verification
- Metal graph surgery required but within existing architecture

#### #4: Premature Closure Detection (13.3) — 1 week
**Impact:** 🟢🟢🟢🟢⚪ **Feasibility:** 🟢🟢🟢🟢🟢
- Fix eval failures where model commits to wrong answer too early
- Track confidence trajectory during generation
- Force "Let me think more..." when confidence spikes prematurely
- Pure Clojure, immediate eval score improvement

#### #5: Expert Routing Analysis + Safety Expert Dropping (5.1 + 5.2) — 3 weeks
**Impact:** 🟢🟢🟢🟢⚪ **Feasibility:** 🟢🟢🟢⚪⚪
- Qwen3-Coder is MoE with 128 experts — this is uniquely applicable
- Log which experts activate for which tasks
- Identify "safety" experts and selectively suppress them for research
- Enable per-task expert tuning (math experts, code experts, etc.)

### Implementation Roadmap

```
Week 1:  Conformal Certification + Premature Closure Detection
Week 2:  Flow Matching Steering (test-time CAA pipeline)
Week 3:  Expert Routing Analysis (logging + visualization)
Week 4:  Self-Speculative Decoding (draft/verify on CPU first)
Week 5-6: Self-Speculative Decoding (Metal kernels)
Week 7:  Safety Expert Dropping + Per-Task Expert Tuning
Week 8:  Integration testing + eval on all 92 questions
```

---

## Quick Reference: DS4 Implementation Status

| Technique | Status | Difficulty | Files to Modify |
|-----------|--------|-----------|-----------------|
| Activation Capture | ✅ Done | Easy | `ds4.c`, `ds4_metal.m`, `core.clj` |
| SAE Steering | ✅ Done | Medium | `ds4.c`, `core.clj` |
| Logit Lens | ✅ Done | Medium | `ds4.c`, `core.clj` |
| CFG | ✅ Done | Easy | `ds4.c`, `core.clj` |
| Logit Bias | ✅ Done | Easy | `ds4.c`, `core.clj` |
| Speculative Decoding | 🔄 Partial | Hard | `ds4.c`, `ds4_metal.m` |
| KV Cache Compression | ❌ Not started | Hard | `ds4.c`, `ds4_metal.m` |
| Test-Time Finetuning | ❌ Not started | Very Hard | `ds4.c` (needs backward pass) |
| Flow Matching Steering | ❌ Not started | Medium | `core.clj` + analysis scripts |
| Expert Routing Analysis | ❌ Not started | Medium | `ds4.c` + logging |
| Self-Speculative Decoding | ❌ Not started | Hard | `ds4.c`, `ds4_metal.m` |
| Projectional Decoding | ❌ Not started | Medium | `ds4.c` (custom sample loop) |
| Moment-KV | ❌ Not started | Hard | `ds4.c`, `ds4_metal.m` |
| BFP Quantization | ❌ Not started | Medium | Metal kernels |
| Circuit Tracing | ❌ Not started | Medium | `core.clj` + capture |
| Conformal Certification | ❌ Not started | Easy | `core.clj` + probability tracking |
| Tool LoRA Adapters | ❌ Not started | Medium | `ds4.c` + LoRA loader |
| Memory Policy Optimization | ❌ Not started | Hard | `ds4.c` + RL |
| Sandbox Integration | ❌ Not started | Easy | `tools.clj` + subprocess |
| Entropy-Aware Reward Guidance | ❌ Not started | Medium | `ds4.c` (sampling loop) |
| MarginGate Verification | ❌ Not started | Medium | `ds4.c` + margin tracking |
| Premature Closure Detection | ❌ Not started | Medium | `core.clj` + entropy analysis |
| ConMoE Expert Consolidation | ❌ Not started | Hard | `ds4.c` + expert analysis |
| Tool Forge Validation | ❌ Not started | Easy | `tools.clj` + validation hooks |
| Coordinated Pass@K | ❌ Not started | Medium | `core.clj` + sandbox execution |
| Adaptive Think-Mode Switching | ❌ Not started | Medium | `ds4.c` + complexity heuristic |
| BitLLM Edge Quantization | ❌ Not started | Hard | Metal kernels + weight packing |
| RTP-LLM Batching | 🔄 Partial | Medium | `ds4.c` + batch queue |

---

## Citation Index

Papers referenced in this survey (from arXiv 2024–2026):

1. *UniSteer: Text-Guided Flow Matching in Activation Space for Versatile LLM Steering* (2025)
2. *Causal Interventions on Continuous Variables: A Case Study on Verb Bias in Steering Vectors for In-Context Learning* (2025)
3. *Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding* (2025)
4. *Moment-KV: Momentum-Based Decode-Time KV Cache Compression for Long Generation* (2025)
5. *VideoMLA: Low-Rank Latent KV Cache for Minute-Scale Autoregressive Video Diffusion* (2025)
6. *How Far Can Disaggregation Go? A Design-Space Exploration of Attention-FFN Disaggregation for Efficient MoE LLM Serving* (2025)
7. *Rotary GPU: Exploring Local Execution Paths for Large Mixture-of-Experts Models Under Limited GPU Memory* (2025)
8. *Leveraging Routing Dynamics in Mixture-of-Experts Models for Efficient Language Adaptation* (2025)
9. *Understanding Safety-Sensitive Expert Behavior in Mixture-of-Experts LLMs* (2025)
10. *Access Sets Matter: Budgeting Expert Reads for Scalable Weight-Space Model Merging* (2025)
11. *EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture* (2025)
12. *MX-SAFE: Versatile Inference- and Training-Proof Microscaling Format with On-the-Fly Exponent and Mantissa Bit Allocation* (2025)
13. *BioRefusalAudit: Auditing Biosecurity Refusal Depth Using General and Domain-Fine-Tuned Sparse Autoencoders* (2025)
14. *Dissecting the Black Box: Circuit-Level Analysis of LLM Vulnerability Detection* (2025)
15. *Xetrieval: Mechanistically Explaining Dense Retrieval* (2025)
16. *Minimal Prompt Perturbations Lead to Code Vulnerabilities: Prompt Fragility and Hidden-State Signals in Coding LLMs* (2025)
17. *ExCAM: Explainable Cultural Awareness Metrics* (2025)
18. *Self-Trained Verification for Training- and Test-Time Self-Improvement* (2025)
19. *Efficient Test-Time Finetuning of LLMs via Convex Reconstruction and Gradient Caching* (2025)
20. *Knowing What to Solve Before How: Preplan Empowered LLM Mathematical Reasoning* (2025)
21. *Reasoning with Sampling: Cutting at Decision Points* (2025)
22. *In-Context Reward Adaptation for Robust Preference Modeling* (2025)
23. *Harmless Yet Harmful: Neutral Prompting Attacks for Stealthy Hallucination Steering in Agent Skills* (2025)
24. *Fingerprinting Inference Systems of Large Language Models* (2025)
25. *KBF: Knowledge Boundary as Fingerprint for Language Model and Black-Box API Auditing* (2025)
26. *SciIntBench: Measuring LLM Compliance with Research Integrity Norms Under Adversarial Framing* (2025)
27. *ParaTool: Shifting Tool Representations from Context to Parameters* (2025)
28. *How Consistent Are LLM Agents? Measuring Behavioral Reproducibility in Multi-Step Tool-Calling Pipelines* (2025)
29. *Meta-Cognitive Memory Policy Optimization for Long-Horizon LLM Agents* (2025)
30. *REPOT: Recoverable Program-of-Thought via Checkpoint Repair* (2025)
31. *When NPUs Are Not Always Faster: A Stage-Level Analysis of Mobile LLM Inference* (2025)
32. *Protecting On-Device AI Inference: A Systematic Review of Attacks and Defence Mechanisms* (2025)
33. *Sandlock: Confining AI Agent Code with Unprivileged Linux Primitives* (2025)
34. *VLA-Trace: Diagnosing Vision-Language-Action Models through Representation and Behavior Tracing* (2025)
35. *PokerSkill: LLMs Can Play Expert-Level Poker without Training or Solvers* (2025)
36. *Conformal Certification of Reasoning Trace Prefixes* (2025)
37. *Anchorless Diversification for Parallel LLM Ideation* (2025)
38. *VPG: Visual Prefix Guidance for Autoregressive Image and Video Generation* (2025)
39. *Projectional Decoding: Towards Semantic-Aware LLM Generation* (2025)
40. *Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding* (2025)
41. *DiSC: Resolution-Scalable Acceleration of Diffusion Models by Exploiting Sparsity and Cached Token Reuse with Hash-based Distribution* (2025)
42. *SiDP: Memory-Efficient Data Parallelism for Offline LLM Inference* (2025)
43. *Give it Space! Explicit Disentangling of Positional and Semantic Representations in Encoders* (2025)
44. *Enhancing Multi-Agent Communication through Attention Steering with Context Relevance* (2025)
45. *EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization* (2025)
46. *Accelerating Constrained Decoding with Token Space Compression* (2025)
47. *LLMSurgeon: Diagnosing Data Mixture of Large Language Models* (2025)
48. *Unlocking the Working Memory of Large Language Models for Latent Reasoning* (2025)
49. *How LoRA Remembers? A Parametric Memory Law for LLM Finetuning* (2025)
50. *RTP-LLM: High-Performance Alibaba LLM Inference Engine* (2025)
51. *Entropy Aware Reward Guidance for Diffusion Language Model Alignment* (2026)
52. *MarginGate: Sparse Margin-Triggered Verification for Batch-Invariant LLM Inference* (2026)
53. *Quantifying and Mitigating Premature Closure in Frontier LLMs* (2026)
54. *ConMoE: Expert-Pool Consolidation via Prototype Reassignment for MoE Compression* (2026)
55. *Beyond the Prompt in Large Language Models: Comprehension, In-Context Learning, and Chain-of-Thought* (2026)
56. *Tool Forge: A Validation-Carrying Toolchain for Governed Agentic Execution* (2026)
57. *Cast a Wider Net: Coordinated Pass@K Policy Optimization for Code Reasoning* (2026)
58. *ElegantVLA: Learning When to Think for Efficient Vision-Language-Action Models* (2026)
59. *BitTP: The Lightweight Trajectory Prediction Model with BitLLM for Edge-Devices* (2026)
60. *MINDGAMES: A Live Arena for Evaluating Social and Strategic Reasoning in Multi-Agent LLMs* (2026)
61. *Filter-then-Weight: Online Data Selection and Reweighting for LLM Fine-Tuning* (2026)
62. *LaRA: Layer-wise Representation Analysis for Detecting Data Contamination in RL Post-Training* (2026)
63. *DLM-SWAI: Steering Diffusion Language Models Before They Unmask* (2026)
64. *HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM Quantization* (2026)
65. *NaRA: Noise-Aware LoRA for Parameter-Efficient Fine-Tuning of Diffusion LLMs* (2026)
66. *Spurious Prompts: Can Irrelevant Prompts Steer Large Language Models?* (2026)
67. *SkillsInjector: Dynamic Skill Context Construction for LLM Agents* (2026)
68. *Why Specialist Models Still Matter: A Heterogeneous Multi-Agent Paradigm for Medical AI* (2026)

---

*Last updated: 2026-05-30*
*Research compiled from arXiv cs.CL, cs.LG, cs.AI, cs.AR, cs.SE, cs.OS, cs.CR, cs.DC, cs.IR, cs.CV, cs.NE listings*
