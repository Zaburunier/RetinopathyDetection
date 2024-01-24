import tensorflow as tf
#from keras.layers.preprocessing import preprocessing_utils as utils
from tensorflow.keras import backend
from keras.engine import base_layer, base_preprocessing_layer


H_AXIS = -3
W_AXIS = -2


class RepeatableRandomTranslation(base_layer.BaseRandomLayer):
    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        **kwargs,
    ):
        base_preprocessing_layer.keras_kpl_gauge.get_cell(
            "RandomTranslation"
        ).set(True)
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.height_factor = height_factor
        if isinstance(height_factor, (tuple, list)):
            self.height_lower = height_factor[0]
            self.height_upper = height_factor[1]
        else:
            self.height_lower = -height_factor
            self.height_upper = height_factor
        if self.height_upper < self.height_lower:
            raise ValueError(
                "`height_factor` cannot have upper bound less than "
                f"lower bound, got {height_factor}"
            )
        if abs(self.height_lower) > 1.0 or abs(self.height_upper) > 1.0:
            raise ValueError(
                "`height_factor` argument must have values between [-1, 1]. "
                f"Received: height_factor={height_factor}"
            )

        self.width_factor = width_factor
        if isinstance(width_factor, (tuple, list)):
            self.width_lower = width_factor[0]
            self.width_upper = width_factor[1]
        else:
            self.width_lower = -width_factor
            self.width_upper = width_factor
        if self.width_upper < self.width_lower:
            raise ValueError(
                "`width_factor` cannot have upper bound less than "
                f"lower bound, got {width_factor}"
            )
        if abs(self.width_lower) > 1.0 or abs(self.width_upper) > 1.0:
            raise ValueError(
                "`width_factor` must have values between [-1, 1], "
                f"got {width_factor}"
            )

        check_fill_mode_and_interpolation(fill_mode, interpolation)

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed

        self.translationValue = None


    def RegenerateTranslationFactors(self, inputs):
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]

        img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)

        height_translate = self._random_generator.random_uniform(
            shape=[batch_size, 1],
            minval=self.height_lower,
            maxval=self.height_upper,
            dtype=tf.float32,
        )
        height_translate = height_translate * img_hd

        width_translate = self._random_generator.random_uniform(
            shape=[batch_size, 1],
            minval=self.width_lower,
            maxval=self.width_upper,
            dtype=tf.float32,
        )
        width_translate = width_translate * img_wd

        self.translationValue = tf.cast(
            tf.concat([width_translate, height_translate], axis=1),
            dtype=tf.float32,
        )


    def call(self, inputs, training=True):
        inputs = convert_inputs(inputs, self.compute_dtype)

        def random_translated_inputs(inputs):
            """Translated inputs with random ops."""
            # The transform op only accepts rank 4 inputs,
            # so if we have an unbatched image,
            # we need to temporarily expand dims to a batch.
            original_shape = inputs.shape
            unbatched = inputs.shape.rank == 3
            if unbatched:
                inputs = tf.expand_dims(inputs, 0)

            output = transform(
                inputs,
                get_translation_matrix(self.translationValue),
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
            )
            if unbatched:
                output = tf.squeeze(output, 0)
            output.set_shape(original_shape)
            return output

        if training:
            return random_translated_inputs(inputs)
        else:
            return inputs


    def compute_output_shape(self, input_shape):
        return input_shape


    def get_config(self):
        config = {
            "height_factor": self.height_factor,
            "width_factor": self.width_factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_translation_matrix(translations, name=None):
    """Returns projective transform(s) for the given translation(s).

    Args:
        translations: A matrix of 2-element lists representing `[dx, dy]`
            to translate for each image (for a batch of images).
        name: The name of the op.

    Returns:
        A tensor of shape `(num_images, 8)` projective transforms
            which can be given to `transform`.
    """
    with backend.name_scope(name or "translation_matrix"):
        num_translations = tf.shape(translations)[0]
        # The translation matrix looks like:
        #     [[1 0 -dx]
        #      [0 1 -dy]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Translation matrices are always float32.
        return tf.concat(
            values=[
                tf.ones((num_translations, 1), tf.float32),
                tf.zeros((num_translations, 1), tf.float32),
                -translations[:, 0, None],
                tf.zeros((num_translations, 1), tf.float32),
                tf.ones((num_translations, 1), tf.float32),
                -translations[:, 1, None],
                tf.zeros((num_translations, 2), tf.float32),
            ],
            axis=1,
        )


def transform(
    images,
    transforms,
    fill_mode="reflect",
    fill_value=0.0,
    interpolation="bilinear",
    output_shape=None,
    name=None,
):
    with backend.name_scope(name or "transform"):
        if output_shape is None:
            output_shape = tf.shape(images)[1:3]
            if not tf.executing_eagerly():
                output_shape_value = tf.get_static_value(output_shape)
                if output_shape_value is not None:
                    output_shape = output_shape_value

        output_shape = tf.convert_to_tensor(
            output_shape, tf.int32, name="output_shape"
        )

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError(
                "output_shape must be a 1-D Tensor of 2 elements: "
                "new_height, new_width, instead got "
                f"output_shape={output_shape}"
            )

        fill_value = tf.convert_to_tensor(
            fill_value, tf.float32, name="fill_value"
        )

        return tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            output_shape=output_shape,
            fill_value=fill_value,
            transforms=transforms,
            fill_mode=fill_mode.upper(),
            interpolation=interpolation.upper(),
        )


def check_fill_mode_and_interpolation(fill_mode, interpolation):
    if fill_mode not in {"reflect", "wrap", "constant", "nearest"}:
        raise NotImplementedError(
            f"Unknown `fill_mode` {fill_mode}. Only `reflect`, `wrap`, "
            "`constant` and `nearest` are supported."
        )
    if interpolation not in {"nearest", "bilinear"}:
        raise NotImplementedError(
            f"Unknown `interpolation` {interpolation}. Only `nearest` and "
            "`bilinear` are supported."
        )


def convert_inputs(inputs, dtype=None):
    if isinstance(inputs, dict):
        raise ValueError(
            "This layer can only process a tensor representing an image or "
            f"a batch of images. Received: type(inputs)={type(inputs)}."
            "If you need to pass a dict containing "
            "images, labels, and bounding boxes, you should "
            "instead use the preprocessing and augmentation layers "
            "from `keras_cv.layers`. See docs at "
            "https://keras.io/api/keras_cv/layers/"
        )
    #inputs = utils.ensure_tensor(inputs, dtype=dtype)
    return inputs