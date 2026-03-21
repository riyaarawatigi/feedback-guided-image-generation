"""
Image Generator Module
Handles Stable Diffusion image generation with reproducible seeds
"""

import torch
from diffusers import StableDiffusionPipeline
import os
from PIL import Image


class ImageGenerator:
    """Wrapper class for Stable Diffusion image generation"""
    
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        """
        Initialize Stable Diffusion pipeline
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading {model_name}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")
        print("Model loaded successfully!")
    
    def generate(self, prompt, output_path, seed=None):
        """
        Generate single image from text prompt
        
        Args:
            prompt: Text description of image
            output_path: Where to save generated image
            seed: Random seed for reproducibility (optional)
            
        Returns:
            PIL.Image: Generated image
        """
        # Set random seed if provided
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            generator = None
        
        # Generate image
        image = self.pipe(
            prompt,
            generator=generator,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save image
        image.save(output_path)
        
        return image


def main():
    """Test function"""
    generator = ImageGenerator()
    test_prompt = "a person and a dog"
    generator.generate(test_prompt, "test_output.png", seed=42)
    print("Test image generated successfully!")


if __name__ == "__main__":
    main()
