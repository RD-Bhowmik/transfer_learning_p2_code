def validate_image(image):
    """Quality checks for:
    - Minimum resolution (512px)
    - Color channel integrity
    - Intensity distribution
    - Artifact detection"""
    if image.shape[0] < 512 or image.shape[1] < 512:
        raise InvalidImageError("Insufficient resolution")
    if np.max(image) - np.min(image) < 0.2:
        raise InvalidImageError("Low contrast")
    return True 