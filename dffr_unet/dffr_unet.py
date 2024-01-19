import keras_preprocessing.image
import tensorflow as tf
from tensorflow import Variable, train, int32, function
from tensorflow.keras.models import Model
from tensorflow.keras.layers import RandomZoom, RandomTranslation
from tensorflow.keras.losses import BinaryCrossentropy as LossBE, Reduction
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.metrics import Mean, AUC, BinaryIoU

import tensorflow_addons as tfa

from tools import FitLogCallback
from dffr_unet import RepeatableRandomZoom
from dffr_unet import RepeatableRandomTranslation
from constants import RANDOM_SEED, IMAGE_SIZE, BATCH_SIZE

LEARNING_RATE = 1e-03
MOMENTUM = 0.9
EPSILON = 1.0e-07
DECAY = 1.0e-07
AMSGRAD = True


class DFFRUNet(Model):
    '''
    Класс для работы с моделью типа MTUnet
    '''
    def __init__(self, *args, **kwargs):
        super(DFFRUNet, self).__init__(*args, **kwargs)

        self.segmentation_loss = LossBE(from_logits = False, name = "segm_loss", reduction = Reduction.SUM)

        self.optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=MOMENTUM, epsilon=EPSILON, amsgrad=AMSGRAD, decay=DECAY)
        self.epoch_counter = Variable(initial_value=0, dtype=int32, trainable=False)
        self.checkpoint = train.Checkpoint(model=self, optimizer=self.optimizer, epoch_counter=self.epoch_counter)
        self.checkpoint_manager = train.CheckpointManager(checkpoint=self.checkpoint, directory="../dffr_unet_checkpoint",
                                                          max_to_keep=40)
        print(f"DFFRUnet last checkpoint: {self.checkpoint_manager.latest_checkpoint}")
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        self.logger = FitLogCallback.FitCallback()
        self.csv_logger = CSVLogger(filename=self.checkpoint_manager.directory + "_training_log.csv", append=True)
        self.lr_logger = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=40, verbose=2,
                                           min_delta=0.01, cooldown=10, min_lr=1.0e-08)

        self.loss_tracker = Mean(name="loss")
        self.od_iou_metric = BinaryIoU(name="od_iou")
        self.us_iou_metric = BinaryIoU(name="us_iou")
        self.ma_ha_iou_metric = BinaryIoU(name="ma_ha_iou")
        self.he_se_iou_metric = BinaryIoU(name="he_se_iou")

        self.reg_loss_tracker = Mean(name = "reg_loss")

        print(f"Initialized DFFRUnet for epoch #{self.epoch_counter.read_value().numpy()}")

        self.zoomLayer = RepeatableRandomZoom.RepeatableRandomZoom(height_factor = (-0.1, 0.25), fill_mode ="constant")
        self.translateLayer = RepeatableRandomTranslation.RepeatableRandomTranslation(0.15, 0.15, fill_mode ='constant')


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.reg_loss_tracker, self.od_iou_metric, self.us_iou_metric, self.ma_ha_iou_metric, self.he_se_iou_metric]


    def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              steps_per_execution=None,
              jit_compile=None,
              **kwargs):
        super(DFFRUNet, self).compile(optimizer = self.optimizer, loss = self.segmentation_loss, metrics = metrics, loss_weights = loss_weights,
                                    weighted_metrics = weighted_metrics, run_eagerly = run_eagerly,
                                    steps_per_execution = steps_per_execution, jit_compile = jit_compile, **kwargs)


    @function(jit_compile=False)
    def train_step(self, data):
        x = data[:, :, :, :3]
        y = data[:, :, :, 3:]

        y = y / 255.0
        y = self.BinarizeMasks(y)
        x = x / 255.0
        x = self.PrepareImage(x)

        x, y = self.AugmentData(x, y)

        # Добавляем отдельную карту для пикселей, которые не содержат никакого маркера заболевания
        # Это позволит софтмаксу складывать сюда весь "мусор"
        unsegmentedPixels = tf.cast(tf.math.logical_not(tf.reduce_any(tf.cast(y, dtype = tf.bool), axis = -1, keepdims=True)), dtype = tf.float32)
        y = tf.concat([y, unsegmentedPixels], -1)

        if (tf.config.functions_run_eagerly()):
            xData = x.numpy()
            yData = y.numpy()
            maData = y[:, :, :, 0].numpy()
            haData = y[:, :, :, 1].numpy()
            heData = y[:, :, :, 2].numpy()
            seData = y[:, :, :, 3].numpy()
            odData = y[:, :, :, 4].numpy()
            usData = y[:, :, :, 5].numpy()
            maHaData = maData + haData
            heSeData = heData + seData
            diseaseMaskData = maData + haData + heData + seData + odData
            #totalMaskData = diseaseMaskData + usData

        with tf.GradientTape() as tape:
            yPred = self(x, training=True)  # Forward pass

            segmLoss = 0

            yMaHa = tf.cast(tf.reduce_any(tf.cast(y[:, :, :, 0:2], dtype=tf.bool), axis=-1, keepdims=True), dtype=tf.float32)
            yHeSe = tf.cast(tf.reduce_any(tf.cast(y[:, :, :, 2:4], dtype=tf.bool), axis=-1, keepdims=True), dtype=tf.float32)

            segmLoss += self.segmentation_loss(yMaHa, yPred[:, :, :, 0]) / BATCH_SIZE
            segmLoss += self.segmentation_loss(yHeSe, yPred[:, :, :, 1]) / BATCH_SIZE
            segmLoss += self.segmentation_loss(y[:, :, :, 4], yPred[:, :, :, 2]) / BATCH_SIZE
            #segmLoss += self.segmentation_loss(y[:, :, :, 5], yPred[:, :, :, 3]) / BATCH_SIZE

            loss = segmLoss
            regLoss = 1e-1 * tf.math.add_n(self.losses)
            loss += regLoss

        if (tf.config.functions_run_eagerly()):
            yPredData = yPred.numpy()
            maHaPredData = yPredData[:, :, :, 0]
            heSePredData = yPredData[:, :, :, 1]
            odPredData = yPredData[:, :, :, 2]
            usPredData = yPredData[:, :, :, 3]

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        if (tf.config.functions_run_eagerly()):
            gradientsData = []
            for grads in gradients:
                if (grads is not None):
                    gradientsData.append(grads.numpy())

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.ma_ha_iou_metric.update_state(yMaHa, tf.expand_dims(yPred[:, :, :, 0], axis = -1))
        self.he_se_iou_metric.update_state(yHeSe, yPred[:, :, :, 1])
        self.od_iou_metric.update_state(y[:, :, :, 4], yPred[:, :, :, 2])
        self.us_iou_metric.update_state(y[:, :, :, 5], yPred[:, :, :, 3])
        self.reg_loss_tracker.update_state(regLoss)

        return {m.name: m.result() for m in self.metrics}


    @function(jit_compile=False)
    def test_step(self, data):
        x = data[:, :, :, :3]
        y = data[:, :, :, 3:]

        y = y / 255.0
        y = self.BinarizeMasks(y)
        x = x / 255.0
        x = self.PrepareImage(x)

        # Добавляем отдельную карту для пикселей, которые не содержат никакого маркера заболевания
        # Это позволит софтмаксу складывать сюда весь "мусор"
        unsegmentedPixels = tf.cast(tf.math.logical_not(tf.reduce_any(tf.cast(y, dtype = tf.bool), axis = -1, keepdims=True)), dtype = tf.float32)
        y = tf.concat([y, unsegmentedPixels], -1)
        #segmentationForLoss = tf.concat([segmentation_ma_ha, segmentation_he_se, tf.expand_dims(y[:, :, :, 4], -1), unsegmentedPixels], -1)

        if (tf.config.functions_run_eagerly()):
            xData = x.numpy()
            yData = y.numpy()
            maData = y[:, :, :, 0].numpy()
            haData = y[:, :, :, 1].numpy()
            heData = y[:, :, :, 2].numpy()
            seData = y[:, :, :, 3].numpy()
            odData = y[:, :, :, 4].numpy()
            usData = y[:, :, :, 5].numpy()
            maHaData = maData + haData
            heSeData = heData + seData

        yPred = self(x, training=False)  # Forward pass

        segmLoss = 0  # self.segmentation_loss(y,  yPred) / BATCH_SIZE #0

        yMaHa = tf.cast(tf.reduce_any(tf.cast(y[:, :, :, 0:2], dtype=tf.bool), axis=-1, keepdims=True), dtype=tf.float32)
        yHeSe = tf.cast(tf.reduce_any(tf.cast(y[:, :, :, 2:4], dtype=tf.bool), axis=-1, keepdims=True), dtype=tf.float32)

        segmLoss += self.segmentation_loss(yMaHa, yPred[:, :, :, 0]) / BATCH_SIZE
        segmLoss += self.segmentation_loss(yHeSe, yPred[:, :, :, 1]) / BATCH_SIZE
        segmLoss += self.segmentation_loss(y[:, :, :, 4], yPred[:, :, :, 2]) / BATCH_SIZE
        #segmLoss += self.segmentation_loss(y[:, :, :, 5], yPred[:, :, :, 3]) / BATCH_SIZE

        loss = segmLoss

        regLoss = 1e-1 * tf.math.add_n(self.losses)
        loss += regLoss

        if (tf.config.functions_run_eagerly()):
            yPredData = yPred.numpy()
            maHaPredData = yPredData[:, :, :, 0]
            #heSePredData = yPredData[:, :, :, 1]

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.ma_ha_iou_metric.update_state(yMaHa, yPred[:, :, :, 0])
        self.he_se_iou_metric.update_state(yHeSe, yPred[:, :, :, 1])
        self.od_iou_metric.update_state(y[:, :, :, 4], yPred[:, :, :, 2])
        self.us_iou_metric.update_state(y[:, :, :, 5], yPred[:, :, :, 3])

        self.reg_loss_tracker.update_state(regLoss)

        return {m.name: m.result() for m in self.metrics}


    @function(jit_compile=False)
    def predict_step(self, data):
        x = data
        x = x / 255.0
        x = self.PrepareImage(x)

        if (tf.config.functions_run_eagerly()):
            xData = x.numpy()

        y = self(x, training=False)  # Forward pass
        if (tf.config.functions_run_eagerly()):
            yData = y.numpy()
            maHaPredData = yData[:, :, :, 0]
            heSePredData = yData[:, :, :, 1]
            odPredData = yData[:, :, :, 2]
            usPredData = yData[:, :, :, 3]

        return y


    @tf.function(jit_compile=False)
    def AugmentData(self, x, y):
        # Сдвиги, зумы и прочие модификации нужно делать одновременно для изображений и масок
        tempXY = tf.concat([x, y], axis=-1)

        # Аугментация 1: Зеркальное отражение по каждой оси
        tempXY = tf.image.random_flip_left_right(tempXY)
        tempXY = tf.image.random_flip_up_down(tempXY)

        # Аугментация 2: масштабирование (зум)
        augmentedX, augmentedY = tempXY[:, :, :, :3], tempXY[:, :, :, 3:]
        self.zoomLayer.RegenerateZoomFactors(BATCH_SIZE)
        self.zoomLayer.fill_value = 0.5
        augmentedX = self.zoomLayer(augmentedX, training=True)
        self.zoomLayer.fill_value = 0.0
        augmentedY = self.zoomLayer(augmentedY, training=True)

        # Аугментация 3: сдвиг
        self.translateLayer.RegenerateTranslationFactors(augmentedX)
        self.translateLayer.fill_value = 0.5
        augmentedX = self.translateLayer(augmentedX, training=True)
        self.translateLayer.fill_value = 0.0
        augmentedY = self.translateLayer(augmentedY, training=True)
        #tempXY = tf.concat([augmentedX, augmentedY], axis=-1)

        # Изменение цветовых параметров нужно делать только для изображений, маски мы хотим получить одни и те же
        #max = tf.reduce_max(tempXY, axis=(1, 2, 3), keepdims=True)
        #min = tf.reduce_min(tempXY, axis=(1, 2, 3), keepdims=True)
        #tempXY = (tempXY - min) / (max - min)
        #augmentedX, augmentedY = tempXY[:, :, :, :3], tempXY[:, :, :, 3:]

        #yFillMask = tf.logical_and(tf.greater(augmentedY, 0.49), tf.less(augmentedY, 0.51))
        #zeros = tf.zeros_like(augmentedY)
        #augmentedY = tf.where(yFillMask, zeros, augmentedY)

        # Аугментация 4: модификация параметров изображения
        imageAugmentRatios = tf.random.uniform((4,), -1.0, 1.0)
        #augmentedX = tf.image.adjust_brightness(augmentedX, .3 * imageAugmentRatios[0])
        #augmentedX = tf.image.adjust_contrast(augmentedX, 1.0 + 0.18 * imageAugmentRatios[1])
        #augmentedX = tf.image.adjust_saturation(augmentedX, 1.0 + 0.18 * imageAugmentRatios[2])
        augmentedX = tf.image.adjust_hue(augmentedX, imageAugmentRatios[3])

        if (tf.config.functions_run_eagerly()):
            sourceXData = x.numpy()
            sourceYData = y.numpy()
            augmentedXYData = tempXY.numpy()
            augmentedXData = augmentedX.numpy()
            augmentedYData = augmentedY.numpy()
            augmentedMaData = augmentedY[:, :, :, 0].numpy()
            augmentedHaData = augmentedY[:, :, :, 1].numpy()
            augmentedHeData = augmentedY[:, :, :, 2].numpy()
            augmentedSeData = augmentedY[:, :, :, 3].numpy()
            augmentedOdData = augmentedY[:, :, :, 4].numpy()

            maMaskData = (augmentedX * tf.expand_dims(augmentedY[:, :, :, 0], axis = -1)).numpy()
            haMaskData = (augmentedX * tf.expand_dims(augmentedY[:, :, :, 1], axis = -1)).numpy()
            heMaskData = (augmentedX * tf.expand_dims(augmentedY[:, :, :, 2], axis = -1)).numpy()
            seMaskData = (augmentedX * tf.expand_dims(augmentedY[:, :, :, 3], axis = -1)).numpy()
            odMaskData = (augmentedX * tf.expand_dims(augmentedY[:, :, :, 4], axis = -1)).numpy()

        return augmentedX, augmentedY


    def BinarizeMasks(self, y):
        segmentation = y

        if (tf.config.functions_run_eagerly()):
            ma = segmentation[:, :, :, 0].numpy()
            ha = segmentation[:, :, :, 1].numpy()
            he = segmentation[:, :, :, 2].numpy()
            se = segmentation[:, :, :, 3].numpy()
            od = segmentation[:, :, :, 4].numpy()

        segmentation = tf.cast(segmentation > 0.0, dtype = tf.float32)

        if (tf.config.functions_run_eagerly()):
            binarizedMa = segmentation[:, :, :, 0].numpy()
            binarizedHa = segmentation[:, :, :, 1].numpy()
            binarizedHe = segmentation[:, :, :, 2].numpy()
            binarizedSe = segmentation[:, :, :, 3].numpy()
            binarizedOd = segmentation[:, :, :, 4].numpy()

        y = segmentation

        return y


    def PrepareImage(self, imgTensor):
        '''
        Подготовка изображения к подаче в нейронную сеть (увеличение контраста и т. д.)
        :param imgTensor: Исходное изображение сетчатки
        :return:
        '''
        preparedImgTensor = 4 * imgTensor - 4 * tfa.image.gaussian_filter2d(imgTensor, 15, 10) + 0.5
        if (tf.config.functions_run_eagerly()):
            imgData = imgTensor.numpy()
            gray = tf.image.adjust_contrast(imgTensor, 0.5)
            grayData = gray.numpy()
            blurred = tfa.image.gaussian_filter2d(imgTensor, 15, 1)
            blurredData = blurred.numpy()
            blurredGray = tf.image.adjust_contrast(blurred, 0.5)
            blurredGrayData = blurredGray.numpy()
            contrast = tf.image.adjust_contrast(imgData, 1.4)
            contrastData = contrast.numpy()
            diffData = 4 * imgData - 4 * blurredData + 0.5

            resultData = preparedImgTensor.numpy()
        return preparedImgTensor


    def CreateUnsegmentedMask(self):
        pass