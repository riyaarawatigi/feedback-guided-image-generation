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

## Models
- Generator: Stable Diffusion 2.1
- Detector: YOLOv8
- Dataset: 30 multi-object prompts (COCO objects only)
