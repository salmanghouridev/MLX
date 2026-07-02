import sys
import os
import time
import numpy as np
from PIL import Image

try:
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.models.sam3.generate import Sam3Predictor
except ImportError:
    print("Error: Missing required packages. Please install them using:")
    print("pip install mlx mlx-vlm pillow numpy")
    sys.exit(1)

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python mlx_sam_bg_remover.py <input_image_path> [text_prompt] [output_image_path]")
        print("Default text_prompt: 'subject'")
        print("Example: python mlx_sam_bg_remover.py photo.jpg 'person' output.png")
        sys.exit(1)

    input_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) >= 3 else "subject"
    
    if len(sys.argv) == 4:
        output_path = sys.argv[3]
    else:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_no_bg_sam3.png"

    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    print(f"Loading image: {input_path}")
    try:
        input_image = Image.open(input_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # 1. Load MLX SAM 3 Model (runs natively on Apple Silicon GPU/Metal)
    model_id = "mlx-community/sam3-4bit"
    print(f"Loading native Apple Silicon MLX model: {model_id}...")
    start_load = time.time()
    try:
        model, processor = load(model_id)
        predictor = Sam3Predictor(model, processor)
    except Exception as e:
        print(f"Error loading MLX model: {e}")
        sys.exit(1)
    print(f"Model loaded in {time.time() - start_load:.2f} seconds.")

    # 2. Perform text-guided segmentation
    print(f"Segmenting image using prompt: '{prompt}'...")
    start_infer = time.time()
    try:
        result = predictor.predict(input_image, text_prompt=prompt)
    except Exception as e:
        print(f"Error during segmentation inference: {e}")
        sys.exit(1)
    print(f"Segmentation inference completed in {time.time() - start_infer:.2f} seconds.")

    # 3. Process masks
    masks = result.masks  # Shape: (N, H, W)
    scores = result.scores

    if len(masks) == 0:
        print(f"No objects matching '{prompt}' were detected. Cannot remove background.")
        sys.exit(1)

    print(f"Found {len(masks)} object(s) matching '{prompt}' with confidence scores: {['{:.2f}'.format(s) for s in scores]}")

    # Combine all masks using logical OR
    combined_mask = np.logical_or.reduce(masks, axis=0)

    # 4. Remove background by adding alpha channel
    image_np = np.array(input_image.convert("RGBA"))
    
    # Scale mask from binary [0, 1] to alpha channel [0, 255]
    alpha_channel = (combined_mask * 255).astype(np.uint8)
    
    # Set the alpha channel of the image
    image_np[..., 3] = alpha_channel

    # 5. Save the output
    print(f"Saving background-removed image to: {output_path}")
    try:
        output_image = Image.fromarray(image_np)
        output_image.save(output_path)
        print("Done!")
    except Exception as e:
        print(f"Error saving output image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
