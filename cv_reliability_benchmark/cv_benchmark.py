#!/usr/bin/env python3
import os
import sys
import time
import json
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def generate_dummy_image(path):
    """Generates a high-quality dummy image with shapes and text if no input image is found."""
    print(f"Creating a synthetic test image at {path}...")
    # Create a 600x600 RGB canvas (white background)
    img = np.ones((600, 600, 3), dtype=np.uint8) * 255
    
    # Draw a blue rectangle (representing a screen/box)
    cv2.rectangle(img, (50, 50), (550, 550), (255, 0, 0), 10)
    
    # Draw a green circle (representing a tennis ball or object)
    cv2.circle(img, (300, 300), (120), (0, 255, 0), -1)
    
    # Draw a red triangle
    pts = np.array([[300, 100], [150, 400], [450, 400]], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 255), 8)
    
    # Add text
    cv2.putText(img, "AI Reliability Test", (100, 520), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    
    cv2.imwrite(path, img)
    return img

def apply_gaussian_noise(image, severity):
    """Adds synthetic Gaussian noise using OpenCV/Numpy."""
    # severity from 1 (low) to 5 (extreme)
    std = severity * 15
    h, w, c = image.shape
    noise = np.random.normal(0, std, (h, w, c)).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_gaussian_blur(image, severity):
    """Applies Gaussian Blur using OpenCV."""
    # severity from 1 (low) to 5 (extreme) -> kernel sizes 3x3 to 21x21
    ksize = severity * 4 + 1  # 5, 9, 13, 17, 21
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_jpeg_compression(image, severity):
    """Simulates compression artifacts by encoding/decoding using OpenCV."""
    # severity from 1 (low compression/high quality) to 5 (high compression/low quality)
    quality = max(2, 100 - severity * 20)  # 80, 60, 40, 20, 2 (extreme)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, 1)

def apply_contrast_degradation(image, severity):
    """Reduces contrast and adjusts brightness using OpenCV."""
    # severity from 1 to 5
    alpha = 1.0 - (severity * 0.15)  # 0.85 down to 0.25 (contrast)
    beta = severity * 10            # slight brightness increase to simulate haze/fog
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def main():
    print("=" * 60)
    print("🚀 OpenCV Computer Vision Reliability Benchmarking Suite")
    print("=" * 60)

    # 1. Setup paths
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_img_path = os.path.join(workspace_root, "bg_remover_using mlx", "beforeimage", "img.jpeg")
    
    # Choose input image
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = default_img_path

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    # Load or generate the image
    if not os.path.exists(img_path):
        print(f"[-] Image not found at target: '{img_path}'")
        img_path = os.path.join(output_dir, "benchmark_sample.jpg")
        image = generate_dummy_image(img_path)
    else:
        print(f"[+] Loading test image: {img_path}")
        image = cv2.imread(img_path)
        if image is None:
            print("[-] Failed to read image with OpenCV. Generating fallback dummy image...")
            img_path = os.path.join(output_dir, "benchmark_sample.jpg")
            image = generate_dummy_image(img_path)

    # Convert to RGB for visualization/PyTorch (OpenCV loads BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Load the Pre-trained PyTorch Model
    print("\n[+] Loading pre-trained MobileNetV3-Small classifier...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Targeting device: {device.type.upper()}")

    try:
        # Load weights and model
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        categories = weights.meta["categories"]
        model = model.to(device)
        model.eval()
        print("[+] Model loaded successfully from torchvision.")
    except Exception as e:
        print(f"[-] Error loading model or weights: {e}")
        print("[-] Switching to offline architecture fallback...")
        model = models.mobilenet_v3_small(pretrained=False)
        categories = [f"class_{i}" for i in range(1000)]
        model = model.to(device)
        model.eval()

    # Preprocessing transform matching ImageNet guidelines
    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Establish baseline inference (Clean Image)
    print("\n[+] Establishing baseline performance with clean image...")
    t_start_prep = time.perf_counter()
    # Profile OpenCV preprocessing
    resized_base = cv2.resize(image_rgb, (256, 256))
    preprocessed_tensor = preprocess_transform(resized_base).unsqueeze(0).to(device)
    t_end_prep = time.perf_counter()
    prep_baseline_ms = (t_end_prep - t_start_prep) * 1000

    # Profile PyTorch inference
    with torch.no_grad():
        t_start_infer = time.perf_counter()
        outputs = model(preprocessed_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        t_end_infer = time.perf_counter()
        infer_baseline_ms = (t_end_infer - t_start_infer) * 1000

    top_prob, top_catid = torch.topk(probabilities, 1)
    baseline_class_idx = top_catid[0].item()
    baseline_class_name = categories[baseline_class_idx]
    baseline_confidence = top_prob[0].item()

    print(f"    - Baseline Preprocessing: {prep_baseline_ms:.2f} ms")
    print(f"    - Baseline Inference:     {infer_baseline_ms:.2f} ms")
    print(f"    - Top Prediction:         '{baseline_class_name}' (ID: {baseline_class_idx})")
    print(f"    - Confidence Score:       {baseline_confidence * 100:.2f}%")

    # 3. Benchmark Corruptions
    corruptions = {
        "Gaussian Noise": apply_gaussian_noise,
        "Gaussian Blur": apply_gaussian_blur,
        "JPEG Compression": apply_jpeg_compression,
        "Contrast Loss": apply_contrast_degradation
    }

    results = {}

    for corr_name, corr_func in corruptions.items():
        print(f"\n[+] Testing robustness against: {corr_name}")
        results[corr_name] = []
        
        # Test 5 severity levels
        for severity in range(1, 6):
            # Apply OpenCV corruption
            t_start_corr = time.perf_counter()
            corrupted_img = corr_func(image_rgb, severity)
            t_end_corr = time.perf_counter()
            corr_ms = (t_end_corr - t_start_corr) * 1000

            # OpenCV preprocessing
            t_start_prep = time.perf_counter()
            # Perform a standard crop/resize mimicking model entrypoint
            prep_img = cv2.resize(corrupted_img, (256, 256))
            tensor = preprocess_transform(prep_img).unsqueeze(0).to(device)
            t_end_prep = time.perf_counter()
            prep_ms = (t_end_prep - t_start_prep) * 1000

            # Model inference
            with torch.no_grad():
                t_start_infer = time.perf_counter()
                outputs = model(tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                t_end_infer = time.perf_counter()
                infer_ms = (t_end_infer - t_start_infer) * 1000

            # Extract prediction metrics
            top_prob, top_catid = torch.topk(probabilities, 1)
            pred_idx = top_catid[0].item()
            pred_name = categories[pred_idx]
            confidence = top_prob[0].item()

            # Record metrics
            results[corr_name].append({
                "severity": severity,
                "corruption_time_ms": corr_ms,
                "preprocessing_time_ms": prep_ms,
                "inference_time_ms": infer_ms,
                "predicted_class_id": pred_idx,
                "predicted_class_name": pred_name,
                "confidence": confidence,
                "label_drifted": (pred_idx != baseline_class_idx)
            })

            drift_status = "⚠️ DRIFTED" if pred_idx != baseline_class_idx else "✅ STABLE"
            print(f"    Level {severity}: Conf: {confidence*100:5.2f}% | Prep: {prep_ms:4.2f}ms | Infer: {infer_ms:4.2f}ms | Predict: '{pred_name}' [{drift_status}]")

    # 4. Save results to JSON
    json_path = os.path.join(output_dir, "cv_reliability_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "baseline": {
                "class_id": baseline_class_idx,
                "class_name": baseline_class_name,
                "confidence": baseline_confidence,
                "preprocessing_time_ms": prep_baseline_ms,
                "inference_time_ms": infer_baseline_ms
            },
            "benchmarks": results
        }, f, indent=2)
    print(f"\n[+] Saved detailed JSON report to {json_path}")

    # 5. Generate Decay Curves plot using Matplotlib
    print("[+] Plotting reliability decay curves...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Computer Vision Model Reliability Profiling\nModel: MobileNetV3-Small (Inference: {device.type.upper()}) | Input: {os.path.basename(img_path)}", fontsize=14, fontweight='bold')

    # Flat array of subplots
    axs_flat = axs.ravel()
    
    for idx, (corr_name, data_list) in enumerate(results.items()):
        ax = axs_flat[idx]
        severities = [d["severity"] for d in data_list]
        confidences = [d["confidence"] * 100 for d in data_list]
        infer_times = [d["inference_time_ms"] for d in data_list]
        prep_times = [d["preprocessing_time_ms"] for d in data_list]
        drifts = [d["label_drifted"] for d in data_list]

        # Draw baseline line
        ax.axhline(y=baseline_confidence * 100, color='gray', linestyle='--', alpha=0.7, label='Baseline Conf')

        # Plot Confidence
        color = 'tab:blue'
        ax.set_xlabel('Corruption Severity Level')
        ax.set_ylabel('Top Prediction Confidence (%)', color=color)
        
        # Draw scatter points where stable is blue circle, drifted is red X
        for s, c_val, d in zip(severities, confidences, drifts):
            marker = 'X' if d else 'o'
            m_color = 'tab:red' if d else 'tab:blue'
            m_size = 100 if d else 60
            ax.scatter(s, c_val, color=m_color, marker=marker, s=m_size, zorder=5)

        ax.plot(severities, confidences, color=color, linewidth=2, label='Confidence')
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(0, 105)

        # Plot Preprocessing + Inference Latency on twin axis
        ax2 = ax.twinx()
        color_lat = 'tab:orange'
        ax2.set_ylabel('Latency (ms)', color=color_lat)
        total_latency = [p + i for p, i in zip(prep_times, infer_times)]
        ax2.plot(severities, total_latency, color=color_lat, linestyle='-.', marker='s', label='Total Latency')
        ax2.plot(severities, prep_times, color='tab:green', linestyle=':', label='OpenCV Prep')
        ax2.tick_params(axis='y', labelcolor=color_lat)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize='small')

        ax.set_title(corr_name, fontweight='semibold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "reliability_decay_curves.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"[+] Saved reliability decay curves chart to: {chart_path}")
    print("=" * 60)
    print("🎯 Benchmarking complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
