"""
Vanilla Baseline
Generates one image per prompt with no modifications
"""

import pandas as pd
import os
import sys

# Allow imports from same directory
sys.path.append(os.path.dirname(__file__))
from generator import ImageGenerator


def run_vanilla_baseline(prompts_csv='data/prompts.csv', 
                         output_dir='generated_images/vanilla',
                         results_csv='results/vanilla_results.csv'):
    """
    Run vanilla baseline experiment
    
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
    print("RUNNING VANILLA BASELINE")
    print("=" * 60)
    
    # Process each prompt
    for idx, row in prompts_df.iterrows():
        prompt_id = row['prompt_id']
        prompt_text = row['prompt_text']
        
        print(f"\n[{idx+1}/{len(prompts_df)}] Prompt {prompt_id}: {prompt_text}")
        
        # Generate image
        output_path = f"{output_dir}/prompt_{prompt_id:03d}.png"
        generator.generate(prompt_text, output_path, seed=42+prompt_id)
        
        # Record result
        results.append({
            'prompt_id': prompt_id,
            'prompt_text': prompt_text,
            'method': 'vanilla',
            'image_path': output_path,
            'num_generated': 1
        })
        
        print(f"  Saved: {output_path}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    results_df.to_csv(results_csv, index=False)
    
    print("\n" + "=" * 60)
    print(f"VANILLA BASELINE COMPLETE")
    print(f"Generated {len(results)} images")
    print(f"Results saved to: {results_csv}")
    print("=" * 60)


if __name__ == "__main__":
    run_vanilla_baseline()
