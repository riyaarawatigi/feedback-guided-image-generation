

````markdown
# Object Omission Mitigation for Text-to-Image Generation

Implementation of detection-guided feedback refinement for improving compositional image generation with Stable Diffusion.

## Overview

Text-to-image models often fail to generate all requested objects in multi-object prompts. This project uses YOLOv8 object detection to identify missing objects and iteratively refines prompts until all requested objects appear.

## Results

- **83.3% success rate** (vs **23.3% baseline**, **46.7% rejection sampling**)
- **1.78× improvement** with **15% fewer generations**
- Works **without model retraining**

## Installation

```bash
git clone https://github.com/riyaarawatigi/feedback-guided-image-generation.git
cd feedback-guided-image-generation
pip install -r requirements.txt
````

## Quick Start

```bash
python scripts/generate.py --prompt "a bicycle next to a car"
```

## Usage

### Single Image Generation

```bash
python scripts/generate.py --prompt "your prompt here" --seed 42
```

### Run Feedback-Guided Refinement

```bash
python scripts/run_feedback.py --prompt "a bicycle next to a car" --max-attempts 9
```

### Run Baselines

```bash
python scripts/run_baselines.py
```

### Reproduce Paper Results

```bash
python scripts/run_all_experiments.py
```

### Summarize Results

```bash
python scripts/summarize_results.py
```

### Generate Result Plots

```bash
python scripts/plot_results.py
```

## Requirements

* Python 3.8+
* CUDA-capable GPU recommended
* See `requirements.txt` for full dependencies

## Project Structure

```text
object-omission-mitigation/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── config.yaml
├── data/
│   └── prompts.csv
├── results/
│   ├── feedback_realtime_results.csv
│   ├── rejection_5_results_CLEAN.csv
│   ├── vanilla_detection.csv
│   └── structured_detection.csv
├── notebooks/
│   └── MMFM_TESTING2_clean.ipynb
├── src/
│   └── object_omission/
│       ├── __init__.py
│       ├── generator.py
│       ├── detection.py
│       ├── feedback.py
│       ├── baselines.py
│       ├── helpers.py
│       └── metrics.py
├── scripts/
│   ├── generate.py
│   ├── run_feedback.py
│   ├── run_baselines.py
│   ├── run_all_experiments.py
│   ├── summarize_results.py
│   └── plot_results.py
└── assets/
    └── example_outputs/
```

## Results Summary

| Method          | Success Rate | Avg Generations |
| --------------- | ------------ | --------------- |
| Vanilla         | 23.3%        | 1.00            |
| Rejection k=5   | 46.7%        | 5.00            |
| Feedback (Ours) | 83.3%        | 4.23            |

## Citation

```bibtex
@article{paul2024object,
  title={Object Omission Mitigation for Compositional Image Generation via Detection-Guided Refinement},
  author={Paul, Pranjaly and Savran, Rumeysa and Arawatagi, Riya},
  year={2024}
}
```

## License

MIT License - see `LICENSE` file.

## Authors

* **Pranjaly Paul** - [pranjaly.paul@utn.de](mailto:pranjaly.paul@utn.de)
* **Rumeysa Savran** - [rumeysa.savran@utn.de](mailto:rumeysa.savran@utn.de)
* **Riya Arawatagi** - [riya.arawatagi@utn.de](mailto:riya.arawatagi@utn.de)

Technische Universität Nürnberg

## Acknowledgments

Course project for **Multimodal Foundation Models**. Built using **Stable Diffusion**, **Diffusers**, and **YOLOv8**.

```
```
