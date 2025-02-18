def normalize_image(image, method="zscore"):
    """
    Standardizes image intensities using:
    - Z-score normalization (μ=0, σ=1)
    - Min-max scaling [0-1]
    - RGB preservation mode
    """
    if method == "tanh":
        # Scale to [-1, 1] range
        image = 2.0 * (image - np.min(image)) / (np.max(image) - np.min(image)) - 1.0
        return image
    if method == "zscore":
        image = (image - np.mean(image)) / np.std(image)
    elif method == "minmax":
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def resize_image(image, target_size=(512, 512)):
    """Maintains aspect ratio with bilinear interpolation"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR) 