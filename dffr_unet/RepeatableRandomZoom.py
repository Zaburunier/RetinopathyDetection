import tensorflow as tf
#from keras.layers.preprocessing import preprocessing_utils as utils
from tensorflow.keras import backend
from keras.engine import base_layer, base_preprocessing_layer


H_AXIS = -3
W_AXIS = -2


# Реализация RandomZoom, переработанная под многократный вызов с одним и тем же случайным показателем зума
class RepeatableRandomZoom(base_layer.BaseRandomLayer):
    def __init__(
        self,
        height_factor,
        width_factor=None,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        **kwargs,
    ):
        base_preprocessing_layer.keras_kpl_gauge.get_cell("RandomZoom").set(
            True
        )
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.height_factor = height_factor
        if isinstance(height_factor, (tuple, list)):
            self.height_lower = height_factor[0]
            self.height_upper = height_factor[1]
        else:
            self.height_lower = -height_factor
            self.height_upper = height_factor

        if abs(self.height_lower) > 1.0 or abs(self.height_upper) > 1.0:
            raise ValueError(
                "`height_factor` argument must have values between [-1, 1]. "
                f"Received: height_factor={height_factor}"
            )

        self.width_factor = width_factor
        if width_factor is not None:
            if isinstance(width_factor, (tuple, list)):
                self.width_lower = width_factor[0]
                self.width_upper = width_factor[1]
            else:
                self.width_lower = -width_factor
                self.width_upper = width_factor

            if self.width_lower < -1.0 or self.width_upper < -1.0:
                raise ValueError(
                    "`width_factor` argument must have values larger than -1. "
                    f"Received: width_factor={width_factor}"
                )

        check_fill_mode_and_interpolation(fill_mode, interpolation)

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed

        self.zoomValue = None


    def RegenerateZoomFactors(self, batch_size):
        height_zoom = self._random_generator.random_uniform(
            shape=[batch_size, 1],
            minval=1.0 + self.height_lower,
            maxval=1.0 + self.height_upper,
        )
        if self.width_factor is not None:
            width_zoom = self._random_generator.random_uniform(
                shape=[batch_size, 1],
                minval=1.0 + self.width_lower,
                maxval=1.0 + self.width_upper,
            )
        else:
            width_zoom = height_zoom

        self.zoomValue = tf.cast(
            tf.concat([width_zoom, height_zoom], axis=1), dtype=tf.float32
        )


    def call(self, inputs, training=True):
        inputs = convert_inputs(inputs, self.compute_dtype)

        def random_zoomed_inputs(inputs):
            """Zoomed inputs with random ops."""
            original_shape = inputs.shape
            unbatched = inputs.shape.rank == 3
            # The transform op only accepts rank 4 inputs,
            # so if we have an unbatched image,
            # we need to temporarily expand dims to a batch.
            if unbatched:
                inputs = tf.expand_dims(inputs, 0)

            inputs_shape = tf.shape(inputs)
            img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
            img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)

            output = transform(
                inputs,
                get_zoom_matrix(self.zoomValue, img_hd, img_wd),
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
                interpolation=self.interpolation,
            )
            if unbatched:
                output = tf.squeeze(output, 0)
            output.set_shape(original_shape)
            return output

        if training:
            return random_zoomed_inputs(inputs)
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


def get_zoom_matrix(zooms, image_height, image_width, name=None):
    """Returns projective transform(s) for the given zoom(s).

    Args:
        zooms: A matrix of 2-element lists representing `[zx, zy]`
            to zoom for each image (for a batch of images).
        image_height: Height of the image(s) to be transformed.
        image_width: Width of the image(s) to be transformed.
        name: The name of the op.

    Returns:
        A tensor of shape `(num_images, 8)`. Projective transforms which can be
            given to operation `image_projective_transform_v2`.
            If one row of transforms is
            `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps the *output* point
            `(x, y)` to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
            where `k = c0 x + c1 y + 1`.
    """
    with backend.name_scope(name or "zoom_matrix"):
        num_zooms = tf.shape(zooms)[0]
        # The zoom matrix looks like:
        #     [[zx 0 0]
        #      [0 zy 0]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Zoom matrices are always float32.
        x_offset = ((image_width - 1.0) / 2.0) * (1.0 - zooms[:, 0, None])
        y_offset = ((image_height - 1.0) / 2.0) * (1.0 - zooms[:, 1, None])
        return tf.concat(
            values=[
                zooms[:, 0, None],
                tf.zeros((num_zooms, 1), tf.float32),
                x_offset,
                tf.zeros((num_zooms, 1), tf.float32),
                zooms[:, 1, None],
                y_offset,
                tf.zeros((num_zooms, 2), tf.float32),
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
    """Applies the given transform(s) to the image(s).

    Args:
        images: A tensor of shape
            `(num_images, num_rows, num_columns, num_channels)` (NHWC).
            The rank must be statically known
            (the shape is not `TensorShape(None)`).
        transforms: Projective transform matrix/matrices.
            A vector of length 8 or tensor of size N x 8.
            If one row of transforms is [a0, a1, a2, b0, b1, b2,
            c0, c1], then it maps the *output* point `(x, y)`
            to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
            `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to the
            transform mapping input points to output points.
            Note that gradients are not backpropagated
            into transformation parameters.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode
            (one of `{"constant", "reflect", "wrap", "nearest"}`).
        fill_value: a float represents the value to be filled outside
            the boundaries when `fill_mode="constant"`.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        output_shape: Output dimension after the transform, `[height, width]`.
            If `None`, output is the same size as input image.
        name: The name of the op.

    Fill mode behavior for each valid value is as follows:

    - `"reflect"`: `(d c b a | a b c d | d c b a)`
    The input is extended by reflecting about the edge of the last pixel.

    - `"constant"`: `(k k k k | a b c d | k k k k)`
    The input is extended by filling all
    values beyond the edge with the same constant value k = 0.

    - `"wrap"`: `(a b c d | a b c d | a b c d)`
    The input is extended by wrapping around to the opposite edge.

    - `"nearest"`: `(a a a a | a b c d | d d d d)`
    The input is extended by the nearest pixel.

    Input shape:
        4D tensor with shape: `(samples, height, width, channels)`,
            in `"channels_last"` format.

    Output shape:
        4D tensor with shape: `(samples, height, width, channels)`,
            in `"channels_last"` format.

    Returns:
        Image(s) with the same type and shape as `images`, with the given
        transform(s) applied. Transformed coordinates outside of the input image
        will be filled with zeros.
    """
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