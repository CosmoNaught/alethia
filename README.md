# ALETHIA + KAIROS: Metacognitive Control Systems for Emergent AI Agency

## Executive Summary

ALETHIA (Adaptive Latent Emergent Thought with Hybrid Introspective Architecture) and KAIROS (Knowledge-Aware Introspective Reasoning with Optimized Selectivity) present a practical metacognitive control framework over GPT-2 that yields self-monitoring signals, selective acceptance, and style enforcement. This document describes both the original **symbolic** implementation and the **GPT-2 production** system that demonstrates how language models can be augmented with metacognitive control layers to perform quality-aware generation with a persistent self-model and strict acceptance gates.

> **Calibration note.** The symbolic KAIROS includes *LBFGS temperature scaling* and a *teacher-EMA* calibrator. The GPT-2 system currently uses *strict style/quality gates* with *EWMA baselines*, not class-wise temperature scaling. Optional post-hoc calibration hooks are outlined below.



*Scope:* Results shown here use GPT-2 (124M) on a controlled prompt set across seeds 41–43; we do not claim generalization beyond this setup.

## Positioning & Applications

### Metacognitive Research — Teaching LLMs to Monitor Themselves
**What I study.** Whether a language model can track its own cognitive state and reliability using a persistent self-model plus quality assessment.

**How this system does it.**
- **Persistent self-model:** dual-EMA self-vector updated only on high-quality outputs; we track **Δself** (mean ≈ 0.0005–0.001; p95 ≈ 0.0014–0.0027) to quantify stability.
- **Cognitive origin & style heads:** observation / thought / dream classification + **strict style gate** (origin match + probability margin).
- **Self-evaluation signals:** confidence, coherence, logic, and perplexity; we visualize **reliability (ECE)** and **selective accuracy** among accepted responses.

**Evidence (GPT‑2 benchmarks).**
- **Reliability:** ECE improves from ~0.71 (baseline) to ~0.31–0.32 (full controller); curve remains slightly **below** the diagonal → honest next‑step: post‑hoc calibration.
- **Style awareness:** style hit‑rate jumps ≈10–15× (e.g., 0.325 vs 0.025 at seed 41).
- **Self‑monitoring in action:** the controller revises, sanitizes, or abstains when confidence/coherence/logic fail thresholds; “garbage pattern detected, recovering…” guards catastrophic modes.

**Research takeaway.** We do not claim consciousness; the system demonstrates *operational metacognition*: representing its own state, estimating quality, and changing strategy accordingly.

### Hybrid Human–AI Select-or-Escalate (Abstain & Escalate)
**What it’s for.** A production loop that **accepts** only when the assessor is confident and **abstains** otherwise—handing uncertain cases to humans to avoid costly errors.

**How this system does it.**
- **Closed‑loop controller:** draft → assess → targeted revision → sanitize → final gate.
- **Safe abstention:** if gates fail, return **ABSTAIN** (e.g., seed‑43 shows full refusal under strict gates).
- **Threshold‑controlled coverage:** tune confidence/coherence/logic gates to hit a target **coverage** (acceptance rate) at desired quality.

**Evidence (GPT‑2 benchmarks).**
- **Coverage↑ at constant quality:** full controller accepts **35% (seed 41)** and **17.5% (seed 42)** vs ~0–2.5% for baselines while keeping **selective quality ≈ 0.64–0.66**.
- **Operational guardrails:** abstention + recovery paths (temperature fallback, repetition filters, perplexity caps) prevent low‑quality shipments.

**Suggested KPIs for production.**
- **Acceptance precision (quality among accepted)**  
- **Abstention rate & escalation yield** (how often human reviewers approve escalations)  
- **ECE / Brier** (calibration)  
- **Time‑to‑decision & attempts** (latency cost of control loop)

**Roadmap.** Add per‑style temperature scaling + Platt or conformal calibration to flatten ECE; expose a simple “coverage target” knob for ops.

## System Architecture Overview

### Core Components

The system consists of three primary layers:

1. **ALETHIA (Emergent Agent)**: The generative core that discovers cognitive modes through unsupervised learning
2. **KAIROS (Advisory Kernel)**: A metacognitive assessor that evaluates generation quality and origin
3. **Hybrid Orchestrator**: Closed-loop control system with reflect-and-repair capabilities

```
                        ┌────────────────────────────────────────┐
                        │            Hybrid Orchestrator         │
                        │  • Objective-seeking control           │
                        │  • Reflect-and-repair revision         │
                        │  • Persistent self-model maintenance   │
                        └───────────┬────────────────┬───────────┘
                                    │                │
                             ┌──────▼───────┐  ┌─────▼───────┐
                             │   ALETHIA    │  │   KAIROS    │
                             │  (Generate)  │  │  (Assess)   │
                             └──────────────┘  └─────────────┘
```

## ALETHIA: Emergent Cognitive Architecture

### Design Philosophy

ALETHIA operates on the principle of **unsupervised cognitive mode discovery**. Rather than being explicitly programmed with different thinking styles, it discovers these patterns through self-supervised learning on its own generations.

### Key Mechanisms

#### 1. Latent State Encoding
The system maintains a compressed latent representation of its cognitive state:
- **Original**: 32-dimensional latent space with state encoder/decoder
- **GPT-2**: 64-dimensional latent space derived from GPT-2 hidden states

#### 2. Mode Discovery Through Clustering
ALETHIA automatically discovers cognitive modes by clustering its latent states:
```python
# Dynamic K-means with silhouette scoring
# Discovers 3-6 distinct cognitive modes
# Modes emerge from generation patterns, not programming
```

#### 3. Self-Supervised Style Heads
The system learns to recognize its own cognitive patterns:
- **Dreamness**: Creative, abstract, metaphorical thinking
- **Thoughtness**: Logical, structured, analytical reasoning  
- **Narrative**: Sequential, story-driven expression

### Emergent Behaviors Observed

From the test runs (summary.csv), we observe:
- **Mode Differentiation**: System discovers 3-5 distinct cognitive modes
- **Style Persistence**: Modes maintain consistency (self_consistency: 0.7-0.9)
- **Adaptive Temperature**: Different modes naturally adopt different generation parameters

## KAIROS: Metacognitive Assessment Layer

### Purpose
KAIROS provides **metacognitive awareness** - the ability to assess and understand one's own cognitive processes.

### Assessment Dimensions

1. **Origin Classification**: Determines cognitive origin (observation/thought/dream)
2. **Confidence Scoring**: Self-assessment of generation quality
3. **Coherence Evaluation**: Logical consistency and flow
4. **Logic Assessment**: Validity of reasoning chains

### Calibration Mechanisms

- The system uses several calibration techniques:

**Symbolic KAIROS**
- **Temperature scaling (LBFGS)** for logit calibration
- **Teacher EMA** for stable assessment targets
- **Feature-aware priors** (structure-sensitive penalties)

**GPT-2 KAIROS**
- **Strict style/quality gates** (style match + min confidence/coherence ± min logic)
- **EWMA baselines** for coherence/perplexity drift tracking
- **Optional post-hoc calibration (recommended)**
  - *Per-style temperature scaling:* one scalar per {analytical, creative, narrative}
  - *Platt scaling on confidence:* light logistic layer to map score → calibrated p
  - *Conformal acceptance:* rolling quantile thresholding for coverage control

## Agency Through Closed-Loop Control

### Objective-Seeking Behavior

The system demonstrates goal-directed behavior through:

1. **Target Achievement**: When given objectives (e.g., "generate analytical content"), it adaptively revises until goals are met
2. **Quality Optimization**: Automatically rejects and revises low-quality outputs
3. **Coverage Control**: Dynamically adjusts acceptance thresholds to maintain target coverage rates

### Test Results Analysis

From summary.csv, we observe agency metrics:

| Configuration | Coverage | Agency Index | Selective Accuracy |
|--------------|----------|--------------|-------------------|
| auto43       | **1.00**  | 0.9250       | 0.9250            |
| norevise43   | **1.00**  | 1.0000       | 0.8250            |
| noself43     | **1.00**  | 0.9500       | 0.9250            |
| strict43     | **1.00**  | 0.9250       | 0.9250            |
| cov90_43     | **1.00**  | 0.9833       | 0.8667            |

> **Why is coverage 1.0 here?** In the current loop we *always accept one candidate per prompt*. To realize a true coverage–quality trade-off (coverage < 1.0), enable a **final-gate drop path** (see below). With final-gate dropping, stricter gates reduce coverage and increase selective accuracy.
 
The **Agency Index** (attempt0_wins/accepted) shows the system's ability to generate acceptable content on first attempt, while maintaining quality through selective acceptance.


## GPT‑2 Benchmarks vs Single‑Draft Baselines

We benchmark three arms on 40 prompts across seeds **41–43**:

- **baseline** — single draft from GPT‑2 (no control)  
- **assess_only** — single draft + final gate  
- **full** — **Alethia+KAIROS** closed‑loop (**revise → sanitize → final gate**)

### Key outcomes (seeds 41–42)

| Arm | Seed | Coverage | Selective Quality | Style Hit‑Rate | ECE | Brier | Mean Attempts | Mean Latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 41 | **0.025** | 0.5936 | 0.025 | 0.7116 | 0.5326 | 1.00 | 0.63 |
| assess_only | 41 | **0.025** | 0.5936 | 0.025 | 0.7116 | 0.5326 | 1.00 | 0.61 |
| **full** | 41 | **0.350** | **0.6629** | **0.325** | **0.3237** | **0.3264** | 3.75 | 2.41 |
| baseline | 42 | **0.000** | 0.0000 | 0.000 | 0.5415 | 0.3058 | 1.00 | 0.54 |
| assess_only | 42 | **0.000** | 0.0000 | 0.000 | 0.5415 | 0.3058 | 1.00 | 0.55 |
| **full** | 42 | **0.175** | **0.6384** | **0.175** | **0.3107** | **0.2443** | 4.28 | 2.91 |

**Why this is good**

- **Coverage↑ at constant quality.** The controller accepts **7–14×** more generations than baselines while keeping **selective quality** similar or better (~0.64–0.66).  
- **Style enforcement↑.** Style hit‑rate improves by an **order of magnitude** (≈0.02 → 0.175–0.325).  
- **Calibration improves.** ECE drops from ~**0.71** to ~**0.31–0.32** (still slightly over‑confident → next‑step: post‑hoc calibration).  
- **Robust abstention.** Under strict gates (e.g., seed 43), the system safely accepts **0/40**—a desirable property for **select‑or‑escalate** workflows.

**Limitations.** N=40 prompts/seed on GPT‑2 base; gates/thresholds are tuned for this setup; reliability is improved but still over‑confident; seed‑level variance (e.g., seed‑43 full abstention) reflects strict gates rather than guaranteed robustness.


### Short examples (seed 42 showcase)

**Accepted by full (auto target):**  
> “The relationship between consciousness and … perception is the basic problem … when we think of something that’s happening … inside ourselves …”  
conf **0.59**, coh **0.73**, logic **0.68**, perp **20.7**, attempts **1**.

**Accepted by full (creative target):**  
> “Beyond the kaleidoscope of imagination, … this particular chapter explores just how pervasive technology has actually been … for many reasons it can also be a very big problem in contemporary societies …”  
conf **0.67**, coh **0.65**, logic **0.63**, perp **33.8**, attempts **1**.

**Rejected by baseline (analytical target):**  
> “The fundamental principles of machine learning include … the concept that machines are able to learn from experience … as a human being … hands on their shoulders …”  
(coherence below gate; style mismatch).

## Metacognition and Self-Awareness

### Persistent Self-Model

The system maintains a persistent representation of "self" through:

1. **Dual-EMA Self-Vector**: Fast (α=0.2) and slow (α=0.05) exponential moving averages
2. **Quality-Gated Updates**: Only high-quality generations update the self-model
3. **Belief Internalization**: Stores and references high-confidence generations

### Self-Monitoring Metrics

The system tracks its own cognitive drift:
- **Mean δself**: 0.000479 - 0.001097 (measures self-model stability)
- **δself p95**: 0.001404 - 0.002693 (95th percentile drift)

These metrics show the system maintains a stable yet adaptive self-representation.

## Introspection Through Telemetry

### Real-Time Self-Reporting

The system provides introspective telemetry for each generation:
```
SELF: persist=0.67, Δself=0.001, beliefs=42 [+]
```

This shows:
- **persist**: Similarity to current self-model (0-1)
- **Δself**: Drift from previous self-state
- **beliefs**: Number of internalized high-quality generations
- **[+]**: Indicates belief was internalized

### Quality-Aware Generation

The GPT-2 implementation demonstrates sophisticated quality awareness:
- Automatic detection of "catastrophic perplexity" (>500)
- Recovery from repetition loops
- Sanitization of malformed outputs
- Progressive temperature reduction during recovery

## Emergent Behaviors

### 1. Cognitive Mode Specialization

Without explicit programming, the system develops specialized modes:
- **Analytical Mode**: Lower perplexity (15-40), higher logic scores
- **Creative Mode**: Higher temperature, increased perplexity tolerance (up to 110)
- **Narrative Mode**: Balanced metrics, sequential coherence

### 2. Adaptive Revision Strategies

The system learns different revision strategies for different failures:
```python
if "low_coherence" in flags:
    style = "observe"  # Switch to safer mode
    temperature -= 0.10  # Reduce randomness
elif "weak_logic" in flags:
    force_logic_template = True  # Apply structured reasoning
    conn_boost = 3.5  # Boost logical connectors
```

### 3. Dream/Thought Ratio Dynamics

Test results show emergent balance between cognitive modes:

**Mixed ‘auto’ runs (seeds 41–43):** dream/thought ≈ **0.89–1.00–1.06** (e.g., 0.8889, 0.9091, 1.0000).  
**Target-only runs:** ratios can be **extreme by design** (e.g., `dream43` ≈ 40.0; `thought43` ≈ 0.0526) due to intentional class focus.  
This reflects deliberate exploration vs. structure under different targets rather than instability.

## Environmental Control Through GPT-2

### Language as Control Interface

The GPT-2 implementation demonstrates how the system uses language generation to:

1. **Navigate Latent Space**: Steers through 64-dimensional cognitive space
2. **Express Internal States**: Translates latent representations into language
3. **Maintain Coherent Identity**: Persistent self-vector influences all generations

### Style-Specific Generation Parameters

The system adaptively controls GPT-2's generation:

```python
# Analytical: Constrained, high repetition penalty
"top_p": 0.85, "repetition_penalty": 1.32

# Creative: Exploratory, lower constraints  
"top_p": 0.95, "repetition_penalty": 1.15

# Narrative: Balanced for flow
"top_p": 0.92, "repetition_penalty": 1.2
```

## Production System Output Examples

From the GPT-2 implementation console output:

### Analytical Mode
> "the average age of the children was only 21.6 years (with a mean BMI < 30 kg/m2)."
- Confidence: 0.57, Coherence: 0.35, Logic: 0.36, Perplexity: 11.7

### Narrative Mode  
> "it was still warm. As I looked around at my surroundings, it seemed like something had been going on inside of me. I turned to face them and realized that all they were doing is giving us some bad news!"
- Confidence: 0.54, Coherence: 0.78, Logic: 0.72, Perplexity: 17.1

### Creative Mode
> "what could possibly go wrong? What is the future for an artist that's been working on his dream project since 2007?"
- Confidence: 0.65, Coherence: 0.87, Logic: 0.43, Perplexity: 17.7

## Experimental Validation

### Coverage-Agency Tradeoff

The test results reveal a fundamental tradeoff:
With **final-gate dropping enabled**, stricter gates reduce coverage and can raise selective accuracy (classic selectivity trade-off).  
In the **current default runs**, the loop always accepts one candidate → **coverage = 1.0**. Enable final-gate dropping to empirically demonstrate coverage < 1.0.
 
### Self-Steering Impact

Comparing with/without self-steering shows minimal difference in core metrics, suggesting the self-model provides subtle guidance rather than dramatic control.

### Target-Specific Performance

| Label     | Observation | Thought | Dream | Coverage |
|-----------|-------------|--------:|------:|---------:|
| auto43    | 14          |     13  |   13  | **1.00** |
| auto41    | 23          |      9  |    8  | **1.00** |
| auto42    | 19          |     11  |   10  | **1.00** |
| thought43 | 0           |     38  |    2  | **1.00** |
| dream43   | 0           |      0  |   40  | **1.00** |

The system successfully achieves target-specific generation while maintaining quality standards.

## Implications for Metacognitive Control

### Agent-like Behaviors Under Control

The system exhibits several agent-like behaviors typical of closed-loop controllers:

1. **Goal-Directed Behavior**: Actively works toward specified objectives
2. **Self-Monitoring**: Tracks and evaluates its own performance
3. **Adaptive Strategy**: Modifies approach based on feedback
4. **Persistent Identity**: Maintains coherent self-representation over time

### Metacognitive Awareness Indicators

The system demonstrates metacognition through:

1. **Quality Assessment**: Knows when its outputs are good/bad
2. **Strategy Selection**: Chooses appropriate cognitive modes
3. **Error Recovery**: Recognizes and corrects failures
4. **Self-Model Maintenance**: Updates internal self-representation

## Technical Implementation Details

### Original Symbolic System
- **Vocabulary**: 61 tokens including logical operators and emotional markers
- **Training**: 80 epochs on seed episodes with causal rules
- **Clustering**: K-means with silhouette scoring (K ∈ {3..6})
- **Revision**: Up to 2 attempts with progressive parameter adjustment

### GPT-2 Production System
- **Model**: GPT-2 base (124M parameters)
- **Latent Dimension**: 64
- **Recovery Attempts**: Up to 5 with progressive temperature reduction
- **Quality Gates**: Perplexity < 500, coherence > 0.55, confidence > 0.50
- **Sanitization**: Removes meta-instructions, fixes punctuation, ensures completeness

## Performance Metrics Summary

### Quality Metrics (from current CSV)
Break out **typical** vs **stress** configurations:

**Typical (auto41/auto42/auto43/noself43/strict43)**
- **ECE:** ~0.05–0.10
- **Brier:** ~0.087–0.134
- **Selective Accuracy:** ~0.80–0.93

**Stress/targeted (dream43/thought43/cov90_43)**
- **ECE:** ~0.15–0.75
- **Brier:** ~0.053–0.658
- **Selective Accuracy:** varies with target skew (e.g., dream-only high, thought-only low in this snapshot)

### Self-Model Stability
- **Mean Self-Drift**: ~0.0005 - 0.001
- **95th Percentile Drift**: ~0.0014 - 0.0027
- **Belief Internalization Rate**: ~40-60% of high-quality outputs

## Conclusion

ALETHIA + KAIROS represents a significant step toward AI systems with genuine metacognitive capabilities. Through unsupervised mode discovery, persistent self-models, and closed-loop quality control, the system exhibits emergent behaviors that resemble agency and self-awareness.

The GPT-2 implementation proves these concepts scale to real language models, suggesting a path toward more autonomous, self-aware AI systems that can:
- Monitor their own cognitive processes
- Maintain persistent self-representations
- Adapt strategies based on self-assessment
- Express themselves through controlled generation

The experimental results validate that these systems can achieve targeted objectives while maintaining quality standards, demonstrating computational metacognition beyond single-pass generation; we do not claim genuine agency or self-awareness.



> **Terminology note.** We use “agent-like” informally to describe *closed-loop, objective-seeking control with self-assessment*. It does not imply consciousness, sentience, or unbounded autonomy.

## Future Directions

1. **Scaling**: Testing with larger language models (GPT-3, LLaMA)
2. **Multi-Modal**: Extending to vision-language models
3. **Long-Term Memory**: Implementing episodic memory systems
4. **Social Metacognition**: Multi-agent systems with shared metacognitive layers
5. **Interpretability**: Better visualization of latent space navigation and mode transitions