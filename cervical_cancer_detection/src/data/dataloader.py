def create_generator(self, batch_size=32):
    """TF Dataset API integration with:
    - Parallel preprocessing
    - On-the-fly augmentation
    - Automatic resource scaling"""
    return tf.data.Dataset.from_generator(
        self._generator,
        output_signature=(
            tf.TensorSpec(shape=(*self.config.target_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.int32)
        )
    ).prefetch(tf.data.AUTOTUNE) 