# Feedback-Guided Rejection Sampling

Course project investigating iterative prompt refinement for reducing object omission in text-to-image diffusion models.

## Team
- Riya - Image Generation
- Pranjaly - Object Detection
- Rumeysa - Feedback Loop

## What We're Building

A system that automatically detects when AI image generators miss objects and refines prompts to fix the problem.

**Example:**
- Prompt: "a cat, a dog, and a bird"
- Generated: Only cat and dog (bird missing)
- System detects: "bird is missing"
- Refined prompt: "a cat, a dog, and especially a bird"
- Regenerate: Now all three objects present ✓

## Methods
1. Vanilla (1 generation)
2. Structured prompts (1 generation)
3. Rejection sampling (5 or 10 generations)
4. Feedback-guided (our method, ~4 generations)

## Goal
Show that feedback-guided refinement achieves better object recall with fewer total generations than blind rejection sampling.

## Setup
```bash
pip install -r requirements.txt
```

## Run Experiments
```bash
python scripts/baseline_vanilla.py --limit 3
python scripts/baseline_structured.py --limit 3
python scripts/baseline_rejection.py --limit 3
python scripts/feedback_method.py --limit 3
python scripts/analyzer.py
python analysis/create_tables.py
```

The `--limit` flag is useful for smoke tests before running the full 30-prompt experiment.

## Outputs
- Generated images are saved under `generated_images/<method>/`
- Result CSVs are saved under `results/`
- Summary tables are saved under `results/tables/`

## Models
- Generator: Stable Diffusion 2.1
- Detector: YOLOv8
- Dataset: 30 multi-object prompts (COCO objects only)
