# SonicBench: Dissecting the Physical Perception Bottleneck in Large Audio Language Models

*Psychophysically grounded benchmark + toolbox for probing physical audio perception in Large Audio Language Models.*

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx">
    <img src="https://img.shields.io/badge/arXiv-SonicBench-B31B1B.svg" alt="arXiv">
  </a>
  <a href="https://huggingface.co/datasets/YirongSun/SonicBench">
    <img src="https://img.shields.io/badge/HF%20Dataset-SonicBench-16a085.svg" alt="HF Dataset">
  </a>
  <a href="https://github.com/EIT-NLP/SonicBench">
    <img src="https://img.shields.io/github/stars/EIT-NLP/SonicBench?style=social" alt="GitHub Stars">
  </a>
</p>

<p align="center">
  <a href="#tldr">TL;DR</a> •
  <a href="#benchmark-overview">Benchmark</a> •
  <a href="#toolbox">Toolbox</a> •
  <a href="#results">Results</a> •
  <a href="#key-findings">Key Findings</a> •
  <a href="#use-cases">Use Cases</a> •
  <a href="#citation">Citation</a>
</p>

---

## TL;DR

SonicBench is a **psychophysically grounded** benchmark that probes **physical audio perception** rather than semantics:

- **12 core attributes × 5 perceptual dimensions × 2 paradigms (recognition vs. comparison) = 2,400 question-audio pairs.**  
- Existing LALMs/LARMs/OLMs, despite strong semantic and paralinguistic performance, often behave **near chance** and fail to show the expected **human-like advantage on comparison tasks**.  
- Explicit chain-of-thought reasoning brings only **marginal gains**, while **linear probes on frozen encoders reach ≥60% accuracy**, revealing that the main bottleneck lies in **alignment and decoding**.

All dataset files are hosted on **Hugging Face**, and this repo provides:

- The **SonicBench Toolbox** for controlled stimulus generation (`./Toolbox/`), and  
- The **full inference results of 36 systems** (`./Results/`).

---

## Benchmark Overview

### Why Physical Audio Perception?

Large Audio Language Models (LALMs) have recently emerged as a unified interface for a wide range of auditory tasks by aligning pre-trained audio encoders with Large Language Models (LLMs). Existing evaluations, however, **focus predominantly on semantic and paralinguistic capabilities** (ASR, captioning, emotion, etc.), while **systematic evaluation of physical perception** remains limited.

Here, physical perception refers to the ability to interpret **intrinsic properties of audio signals**, such as:

- pitch, loudness, duration,  
- spatial location and reverberation,  
- texture and timbre, and  
- scene-level properties like counting.

These attributes underpin **robust auditory intelligence**. Analogous to how **visual intelligence** grounds complex scene understanding in intrinsic attributes like **color and geometry**, reliable audio reasoning depends on accurate **physical grounding** in basic acoustic attributes. In real-world and embodied settings, for example, an agent must infer urgency or danger from physical cues such as **pitch, tempo, and direction**, even in the absence of semantic content. Without such grounding, strong performance on high-level tasks can easily reflect **dataset shortcuts** rather than genuinely grounded auditory understanding. SonicBench is designed precisely to probe this **physical grounding gap**.


### Tasks, Attributes, and Data

SonicBench targets **12 core physical attributes**, grouped into **5 perceptual dimensions**:

- **Spectral & Amplitude**  
  `pitch`, `brightness`, `loudness`, `velocity`
- **Temporal**  
  `duration`, `tempo`
- **Spatial & Environment**  
  `direction`, `distance`, `reverberation`
- **Timbre**  
  `timbre`, `texture`
- **Scene-Level**  
  `counting`

For each attribute, we define two **complementary psychophysical paradigms**:

1. **Recognition (absolute judgment)**  
   - Input: a **single 4-second** audio clip.  
   - Task: make an **absolute decision** between two physical categories  
     (e.g., “bright” vs. “dark”, “short” vs. “long”, “near” vs. “far”).  
   - Output: `"A"` / `"B"`.

2. **Comparison (relative judgment)**  
   - Input: **two 4-second clips** concatenated with **0.5 seconds of silence** in between (≈ 8.5 seconds total).  
   - Task: make a **relative judgment** of which clip has a larger value along a given attribute  
     (e.g., which is brighter / louder / faster / closer).  
   - Output: `"A"` if the first segment is larger, `"B"` otherwise.

This yields:

> **12 attributes × 2 task types × 100 items = 2,400 question-audio pairs.**

<p align="center">
  <img src="figures/sonicbench_overview.png" width="650" alt="SonicBench Overview">
</p>
<p align="center"><i>
Illustration of SonicBench attributes and perceptual dimensions.
</i></p>

---

## Toolbox

> All toolbox code lives in **`./Toolbox/`**.

The **SonicBench Toolbox** is the core component used in our paper to generate all benchmark stimuli. It allows you to:

- Generate **new controlled audio pairs or singletons** for any of the 12 attributes.  
- Set **attribute-level parameters**.  
- Reproduce or extend the **psychophysical design** of SonicBench in your own experiments.

---

## Results

> All evaluation outputs for the paper are stored under **`./Results/`**.

We release the **full inference results of 36 systems** evaluated on SonicBench. These include:

* Main benchmark accuracy for each model × attribute × task type,
* Detailed **recognition vs. comparison** breakdowns,
* **Encoder probing** results based on `probe_json` splits.

A simplified structure (the exact layout may evolve slightly):

```text
Results/
├── main_results/                      # Main benchmark results for 36 systems
│   ├── qwen3_omni/                    # attributes_task.jsonl - 24 jsonl files
│   ├── ...
├── probing_results/                   # Encoder probing results on probe_json splits
│   ├── qwen3_omni_encoder/            # attributes_task.jsonl - 24 jsonl files
│   ├── ...
└── ...                      
```

---

## Key Findings

SonicBench reveals several consistent patterns across **36 evaluated systems**:
Using SonicBench, we evaluate **36 systems** across three families:

- **LALMs** – Large Audio(-Language) Models built by aligning pre-trained audio encoders with LLMs  
- **LARMs** – audio-specific reasoning models  
- **OLMs** – omni-modal models that include an audio interface

SonicBench uncovers several consistent patterns:

1. **Fundamental physical perception is weak.**  
   Despite strong performance on semantic and paralinguistic benchmarks, most models perform **near random guessing (~50%)** on many SonicBench tasks.  
   Even the best model in our study (Qwen3-Omni) reaches only about **72%** accuracy, far below human performance (~91%).  
   This indicates that current systems often lack **reliable physical grounding**, even when their high-level behavior appears competent.

2. **No human-like advantage on comparison tasks.**  
   In human psychophysics, **relative comparison** is often easier than absolute judgment.  
   In contrast, LALMs and related systems show **no systematic advantage** on comparison tasks;  
   for several attributes, **comparison accuracy is even lower than recognition accuracy**.  
   This suggests that current models struggle with **relational reasoning over physical attributes**.

3. **Inference-time reasoning brings limited gains.**  
   We experiment with **explicit reasoning** and inference-time scaling (longer chain-of-thought, more deliberation).  
   The improvements on SonicBench are **marginal**, indicating that simply adding reasoning tokens cannot compensate for missing or poorly used physical representations.

4. **Encoders perceive more than the full model can use.**  
   When we freeze audio encoders and train **simple linear probes** on `probe_json` splits, these probes consistently achieve **≥60% accuracy** across attributes and, in several cases, **outperform the full end-to-end models**.  
   This shows that the **physical cues are already present** in the encoder representations.  
   The primary bottleneck lies in **alignment and decoding**-the projector and language layers fail to faithfully leverage the sensory information they receive.

For detailed results, please see `./Results/`.

---

## Use Cases

SonicBench is designed primarily as an **evaluation and analysis benchmark** for physical audio perception. Typical use cases include:

* **Benchmarking physical grounding**  
  Evaluate LALMs, LARMs, and OLMs on their ability to perceive core physical attributes.

* **Attribute-wise and dimension-wise diagnostics**  
  Use the 12 attributes and 5 perceptual dimensions to pinpoint which aspects (e.g., spectral vs. spatial vs. scene-level) a model handles well or fails on.

* **Studying recognition vs. comparison behavior**  
  Compare model performance across absolute (recognition) and relative (comparison) paradigms to analyze **relational reasoning** over acoustic signals.

* **Encoder probing and architecture analysis**  
  Use `probe_json` train/eval splits to attach simple probes to audio encoders, isolating where information is lost along the encoder-projector-LLM pipeline.

> We recommend treating all files in `json/` as **held-out test sets**.  
> For training probes or auxiliary models, please use the splits provided under `probe_json/`.

---

## TODO / Roadmap

We are gradually open-sourcing all components used in the SonicBench paper.

- [x] **Benchmark dataset on HuggingFace**  
  `YirongSun/SonicBench` has been uploaded with all audio and JSON files.

- [ ] **arXiv paper link**  
  The SonicBench preprint is being uploaded to arXiv. We will update the badge and citation once the ID is available.

- [ ] **Full inference results under `./Results/`**  
  We are cleaning and organizing the outputs of all evaluated systems and will release them here.

- [ ] **SonicBench Toolbox under `./Toolbox/`**  
  The toolbox used to generate all benchmark stimuli is being refactored and documented. A public, research-friendly version will be pushed to this repository.

---

## Citation

If you find SonicBench useful in your research, please cite:

```bibtex
xxx
```

---

## Contact
```
Email: win1282467298@gmail.com, qiuxinzju@zju.edu.cn, xyshen@eitech.edu.cn  
Organization: EIT-NLP Lab
```
