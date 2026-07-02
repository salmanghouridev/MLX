import sys
import os
import time
from PIL import Image

try:
    from rembg import remove
except ImportError:
    print("Error: 'rembg' is not installed. Please install it using: pip install rembg")
    sys.exit(1)

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python remove_background.py <input_image_path> [output_image_path]")
        print("Example: python remove_background.py photo.jpg result.png")
        sys.exit(1)

    input_path = sys.argv[1]
    
    # If output path is not provided, name it as <input_filename>_no_bg.png
    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_no_bg.png"

    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    print(f"Loading image from: {input_path}")
    try:
        input_image = Image.open(input_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    print("Removing background (this may take a few seconds on the first run to download the U-2-Net model weights)...")
    start_time = time.time()
    
    try:
        # Run background removal (rembg automatically uses the best hardware execution provider on macOS)
        output_image = remove(input_image)
    except Exception as e:
        print(f"Error during background removal: {e}")
        sys.exit(1)
        
    duration = time.time() - start_time
    print(f"Background removed successfully in {duration:.2f} seconds.")

    print(f"Saving output to: {output_path}")
    try:
        output_image.save(output_path)
        print("Done!")
    except Exception as e:
        print(f"Error saving image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
