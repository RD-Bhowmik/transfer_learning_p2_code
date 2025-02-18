class Augmentor:
    def random_rotation(self, image):
        """±15° rotation with border reflection"""
        angle = np.random.uniform(-15, 15)
        return rotate(image, angle, mode='reflect')

    def color_jitter(self, image):
        """Channel-wise intensity variation (5-15%)"""
        jitter = np.random.uniform(0.85, 1.15, 3)
        return np.clip(image * jitter, 0, 1) 