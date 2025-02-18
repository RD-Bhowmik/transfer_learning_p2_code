class Pipeline:
    """
    Multi-stage preprocessing pipeline implementing:
    1. Standardized image normalization
    2. Adaptive contrast enhancement
    3. Data augmentation
    4. Quality validation
    """
    def __init__(self, config):
        self.steps = [
            self._load_image,
            self._validate_dimensions,
            self._normalize,
            self._enhance_contrast,
            self._apply_augmentations,
            self._final_validation
        ]
        self.config = config 

    def process(self, image_path):
        """
        Full processing workflow:
        1. Load → Validate → Normalize → Enhance → Augment → Validate
        2. Maintain EXIF metadata preservation
        3. Automatic quality control gates
        """
        try:
            image = Image.open(image_path)
            for step in self.steps:
                image = step(image)
            return image
        except InvalidImageError as e:
            self.logger.error(f"Rejected image {image_path}: {str(e)}")
            return None 