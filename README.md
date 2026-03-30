# Object Omission Mitigation for Text-to-Image Generation

Implementation of detection-guided feedback refinement for improving compositional image generation with Stable Diffusion.

## Overview

Text-to-image models often fail to generate all requested objects in multi-object prompts. This project uses YOLOv8 object detection to identify missing objects and iteratively refines prompts until all objects appear.

**Results:**
- 83.3% success rate (vs 23.3% baseline, 46.7% rejection sampling)
- 1.78× improvement with 15% fewer generations
- Works without model retraining

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/object-omission-mitigation.git
cd object-omission-mitigation
pip install -r requirements.txt
```

## Quick Start

```bash
python generate.py --prompt "a bicycle next to a car"
```

## Usage

### Single Image Generation
```bash
python generate.py --prompt "your prompt here" --max_attempts 9
```

### Compare Methods
```bash
python run_comparison.py --prompt "a bicycle next to a car"
```

### Reproduce Paper Results
```bash
python experiments/run_all_experiments.py
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (16GB+ recommended)
- See `requirements.txt` for full dependencies

## Project Structure

```
├── generate.py              # Single image generation
├── run_comparison.py        # Compare all methods
├── src/
│   ├── feedback_refinement.py
│   ├── detection.py
│   └── baselines.py
├── experiments/
│   ├── run_all_experiments.py
│   └── prompts.json
└── visualizations/
    └── plot_results.py
```

## Results

| Method | Success Rate | Avg Generations |
|--------|:------------:|:---------------:|
| Vanilla | 23.3% | 1.00 |
| Rejection k=5 | 46.7% | 5.00 |
| **Feedback (Ours)** | **83.3%** | **4.23** |

## Citation

```bibtex
@article{paul2024object,
  title={Object Omission Mitigation for Compositional Image Generation via Detection-Guided Refinement},
  author={Paul, Pranjaly and Savran, Rumeysa and Arawatagi, Riya},
  year={2024}
}
```

## License

MIT License - see LICENSE file

## Authors

- Pranjaly Paul - pranjaly.paul@utn.de
- Rumeysa Savran - rumeysa.savran@utn.de  
- Riya Arawatagi - riya.arawatagi@utn.de

Technische Universität Nürnberg

## Acknowledgments

Course project for Multimodal Foundation Models. Built using Stable Diffusion, Diffusers, and YOLOv8.

