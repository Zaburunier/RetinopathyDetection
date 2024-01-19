import keras_preprocessing.image
import tensorflow as tf
from tensorflow import Variable, train, int32, function
from tensorflow.keras.models import Model
from tensorflow.keras.layers import RandomZoom
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.losses import BinaryCrossentropy as LossBE, CategoricalCrossentropy as LossCE, Reduction
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.metrics import Mean, BinaryCrossentropy as MetricBE, CategoricalCrossentropy as MetricCE, AUC, BinaryIoU

import tensorflow_addons as tfa

from tools import FitLogCallback
from constants import RANDOM_SEED, IMAGE_SIZE, BATCH_SIZE

LEARNING_RATE = 4e-05
MOMENTUM = 0.5
EPSILON = 1.0e-04
DECAY = 1.0e-04
AMSGRAD = True


class MTUnet(Model):
    '''
    Класс для работы с моделью типа MTUnet
    '''
    def __init__(self, *args, **kwargs):
        super(MTUnet, self).__init__(*args, **kwargs)

        self.use_grade_loss = False
        self.grade_loss = LossCE(from_logits = False, name = "grade_loss", reduction = Reduction.SUM)
        self.segmentation_loss = LossBE(from_logits = False, name = "segm_loss", reduction = Reduction.SUM)

        self.optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=MOMENTUM, epsilon=EPSILON, amsgrad=AMSGRAD,
                              decay=DECAY)  # SGD(learning_rate = LEARNING_RATE, momentum = MOMENTUM, nesterov = True, decay = DECAY)#
        self.epoch_counter = Variable(initial_value=0, dtype=int32, trainable=False)
        self.checkpoint = train.Checkpoint(model=self, optimizer=self.optimizer, epoch_counter=self.epoch_counter)
        self.checkpoint_manager = train.CheckpointManager(checkpoint=self.checkpoint, directory="../mtunet_checkpoint",
                                                          max_to_keep=40)
        print(f"MTUNet last checkpoint: {self.checkpoint_manager.latest_checkpoint}")
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        self.logger = FitLogCallback.FitCallback()
        self.csv_logger = CSVLogger(filename=self.checkpoint_manager.directory + "_training_log.csv", append=True)
        self.lr_logger = ReduceLROnPlateau(monitor="val_segm_loss_stats", factor=0.5, patience=40, verbose=2,
                                           min_delta=0.0001, cooldown=10, min_lr=1.0e-08)

        self.loss_tracker = Mean(name="loss_stats")

        self.segm_loss_tracker = Mean(name="segm_loss_stats")
        self.segm_metric = BinaryIoU(name = "segm_iou")
        self.ma_metric = BinaryIoU(name="segm_ma_iou")
        self.ha_metric = BinaryIoU(name="segm_ha_iou")
        self.he_metric = BinaryIoU(name="segm_he_iou")
        self.se_metric = BinaryIoU(name="segm_se_iou")
        self.od_metric = BinaryIoU(name="segm_od_iou")
        self.us_metric = BinaryIoU(name="segm_us_iou")
        self.ma_ha_metric = BinaryIoU(name="segm_ma_ha_iou")
        self.he_se_metric = BinaryIoU(name="segm_he_se_iou")

        self.grade_loss_tracker = Mean(name="grade_loss_stats")
        self.grade_metric = AUC(name="grade_roc_auc")

        self.reg_loss_tracker = Mean(name = "reg_loss_stats")

        print(f"Initialized MTUnet for epoch #{self.epoch_counter.read_value().numpy()}")

        self.zoomLayer = RandomZoom(height_factor = (-.3, .3), fill_mode = "constant")


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.reg_loss_tracker, self.segm_loss_tracker, self.segm_metric, self.od_metric, self.us_metric, self.ma_ha_metric, self.he_se_metric]


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
        #self.optimizer.learning_rate.assign(LEARNING_RATE)
        #self.optimizer.momentum.assign(MOMENTUM)
        #self.optimizer.epsilon = EPSILON
        #self.optimizer.amsgrad = AMSGRAD
        super(MTUnet, self).compile(optimizer = self.optimizer, loss = [self.grade_loss, self.segmentation_loss], metrics = metrics, loss_weights = loss_weights,
                                    weighted_metrics = weighted_metrics, run_eagerly = run_eagerly,
                                    steps_per_execution = steps_per_execution, jit_compile = jit_compile, **kwargs)


    @function(jit_compile=False)
    def train_step(self, data):
        x = data[:, :, :, :3]
        y = data[:, :, :, 3:]
        #currentLayer, y = data
        #currentLayer, y = self.AugmentData(currentLayer, y)
        y = self.BinarizeMasks(y)
        x = x / 255.0
        x = self.PrepareImage(x)
        if (self.use_grade_loss):
            grade, segmentation = y
        else:
            segmentation = y

        # Добавляем отдельную карту для пикселей, которые не содержат никакого маркера заболевания
        # Это позволит софтмаксу складывать сюда весь "мусор"
        unsegmentedPixels = tf.cast(tf.math.logical_not(tf.reduce_any(tf.cast(segmentation, dtype = tf.bool), axis = -1, keepdims=True)), dtype = tf.float32)
        segmentation = segmentation / 255.0
        segmentation = tf.concat([segmentation, unsegmentedPixels], -1)
        #segmentationForLoss = tf.concat([segmentation_ma_ha, segmentation_he_se, tf.expand_dims(segmentation[:, :, :, 4], -1), unsegmentedPixels], -1)
        #segmentationForLoss = tf.concat([segmentation, unsegmentedPixels], -1)
        if (self.use_grade_loss):
            grade = grade / 255.0

        if (tf.config.functions_run_eagerly()):
            xData = x.numpy()
            yData = y.numpy()
            maData = segmentation[:, :, :, 0].numpy()
            haData = segmentation[:, :, :, 1].numpy()
            heData = segmentation[:, :, :, 2].numpy()
            seData = segmentation[:, :, :, 3].numpy()
            odData = segmentation[:, :, :, 4].numpy()
            usData = segmentation[:, :, :, 5].numpy()
            maHaData = maData + haData
            heSeData = heData + seData
            diseaseMaskData = maData + haData + heData + seData + odData
            totalMaskData = diseaseMaskData + usData

        with tf.GradientTape() as tape:
            yPred = self(x, training=True)  # Forward pass
            gradePred, segmentationPred = yPred

            segmLoss = 0 #self.segmentation_loss(segmentation,  segmentationPred) / BATCH_SIZE

            segmentationMaHa = tf.cast(
                tf.reduce_any(tf.cast(segmentation[:, :, :, 0:2], dtype=tf.bool), axis=-1, keepdims=True),
                dtype=tf.float32)
            segmentationHeSe = tf.cast(
                tf.reduce_any(tf.cast(segmentation[:, :, :, 2:4], dtype=tf.bool), axis=-1, keepdims=True),
                dtype=tf.float32)

            segmLoss += self.segmentation_loss(segmentationMaHa, segmentationPred[:, :, :, 0]) / BATCH_SIZE
            segmLoss += self.segmentation_loss(segmentationHeSe, segmentationPred[:, :, :, 1]) / BATCH_SIZE
            segmLoss += self.segmentation_loss(segmentation[:, :, :, 4], segmentationPred[:, :, :, 2]) / BATCH_SIZE
            #segmLoss += self.segmentation_loss(segmentation[:, :, :, 5], segmentationPred[:, :, :, 3]) / BATCH_SIZE

            loss = segmLoss

            regLoss = 2.0 * tf.math.add_n(self.losses)
            loss += regLoss

            if (self.use_grade_loss):
                gradeLoss = 1.0 * self.grade_loss(grade, gradePred)
                loss += gradeLoss

        if (tf.config.functions_run_eagerly()):
            gradePredData = gradePred.numpy()
            segmentationPredData = segmentationPred.numpy()
            maHaPredData = segmentationPredData[:, :, :, 0]
            heSePredData = segmentationPredData[:, :, :, 1]
            odPredData = segmentationPredData[:, :, :, 2]
            usPredData = segmentationPredData[:, :, :, 3]

        # Compute gradients
        #trainable_weights = self.trainable_weights
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
        self.segm_loss_tracker.update_state(segmLoss)
        self.ma_ha_metric.update_state(segmentationMaHa, segmentationPred[:, :, :, 0])
        self.he_se_metric.update_state(segmentationHeSe, segmentationPred[:, :, :, 1])
        self.od_metric.update_state(segmentation[:, :, :, 4], segmentationPred[:, :, :, 2])
        self.us_metric.update_state(segmentation[:, :, :, 5], segmentationPred[:, :, :, 3])

        self.reg_loss_tracker.update_state(regLoss)

        if (self.use_grade_loss):
            self.grade_loss_tracker.update_state(gradeLoss)
            self.grade_metric.update_state(grade, gradePred)

        #self.compiled_metrics.update_state(y, yPred)

        return {m.name: m.result() for m in self.metrics}


    @function(jit_compile=False)
    def test_step(self, data):
        x = data[:, :, :, :3]
        y = data[:, :, :, 3:]
        #currentLayer, y = data
        y = self.BinarizeMasks(y)
        x = x / 255.0
        x = self.PrepareImage(x)
        if (self.use_grade_loss):
            grade, segmentation = y
        else:
            segmentation = y

        # Добавляем отдельную карту для пикселей, которые не содержат никакого маркера заболевания
        # Это позволит софтмаксу складывать сюда весь "мусор"
        unsegmentedPixels = tf.cast(tf.math.logical_not(tf.reduce_any(tf.cast(segmentation, dtype = tf.bool), axis = -1, keepdims=True)), dtype = tf.float32)
        segmentation = segmentation / 255.0
        segmentation = tf.concat([segmentation, unsegmentedPixels], -1)
        #segmentationForLoss = tf.concat([segmentation_ma_ha, segmentation_he_se, tf.expand_dims(segmentation[:, :, :, 4], -1), unsegmentedPixels], -1)
        if (self.use_grade_loss):
            grade = grade / 255.0

        if (tf.config.functions_run_eagerly()):
            xData = x.numpy()
            yData = y.numpy()
            maData = segmentation[:, :, :, 0].numpy()
            haData = segmentation[:, :, :, 1].numpy()
            heData = segmentation[:, :, :, 2].numpy()
            seData = segmentation[:, :, :, 3].numpy()
            odData = segmentation[:, :, :, 4].numpy()
            usData = segmentation[:, :, :, 5].numpy()
            maHaData = maData + haData
            heSeData = heData + seData

        yPred = self(x, training=False)  # Forward pass
        gradePred, segmentationPred = yPred

        segmLoss = 0  # self.segmentation_loss(segmentation,  segmentationPred) / BATCH_SIZE

        segmentationMaHa = tf.cast(tf.reduce_any(tf.cast(segmentation[:, :, :, 0:2], dtype=tf.bool), axis=-1, keepdims=True), dtype=tf.float32)
        segmentationHeSe = tf.cast(tf.reduce_any(tf.cast(segmentation[:, :, :, 2:4], dtype=tf.bool), axis=-1, keepdims=True), dtype=tf.float32)

        segmLoss += self.segmentation_loss(segmentationMaHa, segmentationPred[:, :, :, 0]) / BATCH_SIZE
        segmLoss += 0.2 * self.segmentation_loss(segmentationHeSe, segmentationPred[:, :, :, 1]) / BATCH_SIZE
        segmLoss += 0.08 * self.segmentation_loss(segmentation[:, :, :, 4], segmentationPred[:, :, :, 2]) / BATCH_SIZE
        #segmLoss += self.segmentation_loss(segmentation[:, :, :, 5], segmentationPred[:, :, :, 3]) / BATCH_SIZE

        loss = segmLoss

        regLoss = 2.0 * tf.math.add_n(self.losses)
        loss += regLoss

        if (self.use_grade_loss):
            gradeLoss = 1.0 * self.grade_loss(grade, gradePred)
            loss += gradeLoss

        if (tf.config.functions_run_eagerly()):
            gradePredData = gradePred.numpy()
            segmentationPredData = segmentationPred.numpy()
            maHaPredData = segmentationPredData[:, :, :, 0]
            heSePredData = segmentationPredData[:, :, :, 1]

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.segm_loss_tracker.update_state(segmLoss)
        self.ma_ha_metric.update_state(segmentationMaHa, segmentationPred[:, :, :, 0])
        self.he_se_metric.update_state(segmentationHeSe, segmentationPred[:, :, :, 1])
        self.od_metric.update_state(segmentation[:, :, :, 4], segmentationPred[:, :, :, 2])
        self.us_metric.update_state(segmentation[:, :, :, 5], segmentationPred[:, :, :, 3])

        self.reg_loss_tracker.update_state(regLoss)

        if (self.use_grade_loss):
            self.grade_loss_tracker.update_state(gradeLoss)
            self.grade_metric.update_state(grade, gradePred)

        return {m.name: m.result() for m in self.metrics}


    @function(jit_compile=False)
    def predict_step(self, data):
        x = data
        x = x / 255.0

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


    def SetupSegmentationMode(self):
        '''
        Первичная инициализация весов для сети (если загружаем в первый раз)
        :return:
        '''
        print("Freezing encoder...")

        vgg16Model = VGG16(include_top = False,
                           weights = "imagenet")
        if (self.epoch_counter.read_value() == 0):
            print("Assigning imagenet weights...")

        for i in range(len(vgg16Model.layers) - 1):
            self.layers[i].trainable = False
            if (self.epoch_counter.read_value() == 0):
                self.layers[i].set_weights(vgg16Model.layers[i].get_weights())


    @tf.function(jit_compile = False)
    def AugmentData(self, x, y):
        '''
        Аугментация данных. Проводится здесь, а не в слоях нейросети, т. к. маски сегментации нужно аугментировать синхронно с исходным изображением
        :return:
        '''
        if (self.use_grade_loss):
            grade, segmentation = y
        else:
            segmentation = y

        if (tf.config.functions_run_eagerly()):
            img = x.numpy()
            ma = segmentation[:, :, :, 0].numpy()
            ha = segmentation[:, :, :, 1].numpy()
            he = segmentation[:, :, :, 2].numpy()
            se = segmentation[:, :, :, 3].numpy()
            od = segmentation[:, :, :, 4].numpy()


        tempXY = tf.concat([x, segmentation], axis=-1)

        # Аугментация 1: Зеркальное отражение по каждой оси
        tempXY = tf.image.random_flip_left_right(tempXY)
        tempXY = tf.image.random_flip_up_down(tempXY)

        # Аугментация 2: масштабирование (зум)
        tempXY = self.zoomLayer(tempXY)

        # Аугментация 3: сдвиг
        shiftDirections = tf.random.uniform(shape = (2, ), minval = -.1, maxval = .1)
        tempXY = tfa.image.translate(tempXY, shiftDirections)

        x, segmentation = tempXY[:, :, :, :3], tempXY[:, :, :, 3:]

        # Аугментация 4: Подмена цветовых каналов
        #channelOrder = tf.random.shuffle(tf.constant([0, 1, 2]))
        #currentLayer = tf.stack([currentLayer[:, :, :, channelOrder[0]], currentLayer[:, :, :, channelOrder[1]], currentLayer[:, :, :, channelOrder[2]]], axis = -1)

        if (tf.config.functions_run_eagerly()):
            augmentedImg = x.numpy()
            augmentedMa = segmentation[:, :, :, 0].numpy()
            augmentedHa = segmentation[:, :, :, 1].numpy()
            augmentedHe = segmentation[:, :, :, 2].numpy()
            augmentedSe = segmentation[:, :, :, 3].numpy()
            augmentedOd = segmentation[:, :, :, 4].numpy()

        if (self.use_grade_loss):
            y = grade, segmentation
        else:
            y = segmentation

        return x, y


    @tf.function(jit_compile=True)
    def BinarizeMasks(self, y):
        if (self.use_grade_loss):
            grade, segmentation = y
        else:
            segmentation = y

        if (tf.config.functions_run_eagerly()):
            ma = segmentation[:, :, :, 0].numpy()
            ha = segmentation[:, :, :, 1].numpy()
            he = segmentation[:, :, :, 2].numpy()
            se = segmentation[:, :, :, 3].numpy()
            od = segmentation[:, :, :, 4].numpy()

        segmentation = tf.cast(segmentation > 0.0, dtype = tf.float32) * 255.0

        if (tf.config.functions_run_eagerly()):
            binarizedMa = segmentation[:, :, :, 0].numpy()
            binarizedHa = segmentation[:, :, :, 1].numpy()
            binarizedHe = segmentation[:, :, :, 2].numpy()
            binarizedSe = segmentation[:, :, :, 3].numpy()
            binarizedOd = segmentation[:, :, :, 4].numpy()

        if (self.use_grade_loss):
            y = grade, segmentation
        else:
            y = segmentation

        return y


    @tf.function(jit_compile=True)
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