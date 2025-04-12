import matplotlib.pyplot as plt
import numpy as np
import torch

from expt.geometry import compute_anomaly_map, compute_layer_map, plot_anomaly_map


def test_compute_layer_map():
    # Create sample teacher and student features
    batch_size, channels, height, width = 2, 16, 8, 8
    teacher_features = torch.rand(batch_size, channels, height, width)
    student_features = torch.rand(batch_size, channels, height, width)
    image_size = (32, 32)

    # Compute layer map
    layer_map = compute_layer_map(teacher_features, student_features, image_size)

    # Check shape and properties
    assert layer_map.shape == (batch_size, 1, image_size[0], image_size[1])
    assert torch.all(layer_map >= 0)  # All values should be non-negative


def test_compute_anomaly_map():
    # Create sample features
    batch_size, channels, sizes = 2, 16, [(8, 8), (4, 4), (2, 2)]
    teacher_features = [torch.rand(batch_size, channels, h, w) for h, w in sizes]
    student_features = [torch.rand(batch_size, channels, h, w) for h, w in sizes]
    image_size = (32, 32)

    # Compute anomaly map
    anomaly_map = compute_anomaly_map(teacher_features, student_features, image_size)

    # Check shape and properties
    assert anomaly_map.shape == (batch_size, 1, image_size[0], image_size[1])
    assert torch.all(anomaly_map >= 0)  # All values should be non-negative


def test_plot_anomaly_map():
    # Create sample anomaly maps and original images
    height, width = 32, 32

    # Test case 1: Grayscale image with (1, H, W) anomaly map
    anomaly_map1 = np.random.rand(1, height, width).astype(np.float32)
    original_img1 = np.random.rand(1, height, width).astype(np.float32)

    # Test case 2: RGB image with (B, 1, H, W) anomaly map
    anomaly_map2 = np.random.rand(1, 1, height, width).astype(np.float32)
    original_img2 = np.random.rand(3, height, width).astype(np.float32)

    # Test case 3: Different sized anomaly map and image
    anomaly_map3 = np.random.rand(1, height // 2, width // 2).astype(np.float32)
    original_img3 = np.random.rand(3, height, width).astype(np.float32)

    # Plot anomaly maps
    visualizations = plot_anomaly_map(
        [anomaly_map1, anomaly_map2, anomaly_map3],
        [original_img1, original_img2, original_img3],
    )

    # Check results
    assert len(visualizations) == 3
    for vis in visualizations:
        assert isinstance(vis, np.ndarray)
        assert vis.dtype == np.uint8
        assert len(vis.shape) == 3  # (H, W, 3)
        assert vis.shape[2] == 3  # RGB image

    # Display the visualizations (uncomment for visual debugging)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, vis in enumerate(visualizations):
        axes[i].imshow(vis)
        axes[i].set_title(f"Visualization {i+1}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig("anomaly_map_visualization.png")
    plt.close()


def test_plot_anomaly_map_edge_cases():
    height, width = 32, 32

    # Test with images already in HWC format
    anomaly_map = np.random.rand(height, width).astype(np.float32)  # H, W
    original_img = np.random.rand(height, width, 3).astype(np.float32)  # H, W, C

    visualizations = plot_anomaly_map([anomaly_map], [original_img])
    assert len(visualizations) == 1
    assert visualizations[0].shape[0] == height
    assert visualizations[0].shape[1] == width * 2  # Side-by-side images

    # Test with values > 1 in original image
    anomaly_map = np.random.rand(1, height, width).astype(np.float32)
    original_img = np.random.randint(0, 256, (3, height, width)).astype(np.float32)

    visualizations = plot_anomaly_map([anomaly_map], [original_img])
    assert len(visualizations) == 1

    # Test with flat anomaly map (all zeros)
    anomaly_map = np.zeros((1, height, width), dtype=np.float32)
    original_img = np.random.rand(3, height, width).astype(np.float32)

    visualizations = plot_anomaly_map([anomaly_map], [original_img])
    assert len(visualizations) == 1
