"""
Structured Prompt Baseline
Enhances prompts with explicit formatting before generation
"""

import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))
from generator import ImageGenerator


def structure_prompt(original_prompt, num_objects):
    """
    Transform prompt into structured format
    
    Args:
        original_prompt: Original text prompt
        num_objects: Number of objects in prompt
        
    Returns:
        Structured prompt string
    """
    number_words = {2: "two", 3: "three", 4: "four", 5: "five"}
    num_word = number_words.get(num_objects, str(num_objects))
    
    structured = f"An image containing exactly {num_word} objects: {original_prompt}. Show all {num_word} clearly."
    return structured


def run_structured_baseline(prompts_csv='data/prompts.csv',
                            output_dir='generated_images/structured',
                            results_csv='results/structured_results.csv'):
    """
    Run structured prompt baseline experiment
    
    Args:
        prompts_csv: Path to prompts dataset
        output_dir: Directory to save generated images
        results_csv: Path to save results
    """
    # Load prompts
    prompts_df = pd.read_csv(prompts_csv)
    
    # Initialize generator
    generator = ImageGenerator()
    
    # Storage for results
    results = []
    
    print("=" * 60)
    print("RUNNING STRUCTURED PROMPT BASELINE")
    print("=" * 60)
    
    # Process each prompt
    for idx, row in prompts_df.iterrows():
        prompt_id = row['prompt_id']
        original_prompt = row['prompt_text']
        num_objects = row['num_objects']
        
        # Create structured version
        structured_prompt = structure_prompt(original_prompt, num_objects)
        
        print(f"\n[{idx+1}/{len(prompts_df)}] Prompt {prompt_id}")
        print(f"  Original: {original_prompt}")
        print(f"  Structured: {structured_prompt}")
        
        # Generate image
        output_path = f"{output_dir}/prompt_{prompt_id:03d}.png"
        generator.generate(structured_prompt, output_path, seed=42+prompt_id)
        
        # Record result
        results.append({
            'prompt_id': prompt_id,
            'original_prompt': original_prompt,
            'structured_prompt': structured_prompt,
            'method': 'structured',
            'image_path': output_path,
            'num_generated': 1
        })
        
        print(f"  Saved: {output_path}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    results_df.to_csv(results_csv, index=False)
    
    print("\n" + "=" * 60)
    print(f"STRUCTURED BASELINE COMPLETE")
    print(f"Generated {len(results)} images")
    print(f"Results saved to: {results_csv}")
    print("=" * 60)


if __name__ == "__main__":
    run_structured_baseline()
```

---

# FILE 4: `requirements.txt`
```
torch>=2.0.0
diffusers>=0.25.0
transformers>=4.35.0
accelerate>=0.24.0
pillow>=10.0.0
pandas>=2.0.0
pyyaml>=6.0
