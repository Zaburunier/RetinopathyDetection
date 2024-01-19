from tensorflow.keras.models import Model, Sequential
#from keras.layers import Dense
#from tensorflow.keras.initializers import Zeros, RandomNormal, HeNormal, HeUniform
#from keras.regularizers import L2
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.metrics import Mean, CategoricalAccuracy, Recall, Precision, BinaryAccuracy
from tensorflow.keras.layers import RandomZoom, RandomTranslation
from tensorflow.keras.applications.vgg16 import VGG16

import constants
from constants import RANDOM_SEED, BATCH_SIZE
from tensorflow import train, Variable, int32, function, GradientTape
from tools import FitLogCallback
import time

from tensorflow.keras.optimizers import Adam
#from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras.optimizers import SGD

import tensorflow as tf
import tensorflow_addons as tfa
import keras as keras

LEARNING_RATE = 4e-04
MOMENTUM = 0.9
EPSILON = 1.001e-04
DECAY = 1.001e-04
AMSGRAD = False

class CNNMLP(Model):
    def __init__(self, useBinaryLabels = False, *args, **kwargs):
        super(CNNMLP, self).__init__(*args, **kwargs)

        self.optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=MOMENTUM, epsilon = EPSILON, amsgrad = AMSGRAD)#, decay = DECAY)#SGD(learning_rate = LEARNING_RATE, momentum = MOMENTUM, nesterov = True, decay = DECAY)#
        self.epoch_counter = Variable(initial_value=0, dtype=int32, trainable=False)
        self.checkpoint = train.Checkpoint(model = self, optimizer = self.optimizer, epoch_counter=self.epoch_counter)
        self.checkpoint_manager = train.CheckpointManager(checkpoint = self.checkpoint, directory = "D:\\Study\\Magister\\CourseworkProject\\main\\cnn_mlp_checkpoint", max_to_keep = 10)
        print(f"CNNMLP last checkpoint: {self.checkpoint_manager.latest_checkpoint}")
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        self.logger = FitLogCallback.FitCallback()
        self.csv_logger = CSVLogger(filename=self.checkpoint_manager.directory + "_training_log.csv", append=True)
        self.lr_logger = ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 30, verbose = 2,
                                           min_delta = 0.001, cooldown = 5, min_lr = 1.0e-06)
        self.loss_tracker = Mean(name = "loss")
        self.binaryAccuracy = BinaryAccuracy()
        self.recall = Recall(name = "Recall-binary")
        self.precision = Precision(name = "Precision-binary")
        self.categoricalAccuracy = CategoricalAccuracy()
        self.recallClass0 = Recall(class_id=0, name="Recall-0")
        self.recallClass1 = Recall(class_id=1, name="Recall-1")
        self.recallClass2 = Recall(class_id=2, name="Recall-2")
        self.recallClass3 = Recall(class_id=3, name="Recall-3")
        self.recallClass4 = Recall(class_id=4, name="Recall-4")
        self.precisionClass0 = Precision(class_id=0, name="Precision-0")
        self.precisionClass1 = Precision(class_id=1, name="Precision-1")
        self.precisionClass2 = Precision(class_id=2, name="Precision-2")
        self.precisionClass3 = Precision(class_id=3, name="Precision-3")
        self.precisionClass4 = Precision(class_id=4, name="Precision-4")

        self.zoomLayer = RandomZoom(height_factor = (-0.15, 0.25), fill_mode = "constant", fill_value = 0.5)
        self.translateLayer = RandomTranslation(0.15, 0.15, fill_mode = 'constant', fill_value = 0.5)

        print(f"Initialized CNNMLP for epoch #{self.epoch_counter.read_value().numpy()}")

        self.USE_BINARY_LABELS = useBinaryLabels


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        if self.USE_BINARY_LABELS:
            return [self.loss_tracker, self.binaryAccuracy, self.recall, self.precision]
        else:
            return [self.loss_tracker, self.categoricalAccuracy,
                self.recallClass0, self.recallClass1, self.recallClass2, self.recallClass3, self.recallClass4,
                self.precisionClass0, self.precisionClass1, self.precisionClass2, self.precisionClass3, self.precisionClass4,
                self.binaryAccuracy, self.recall, self.precision]


    #@function
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

        if self.USE_BINARY_LABELS:
            super(CNNMLP, self).compile(optimizer=self.optimizer, loss=loss, metrics=metrics + [self.binaryAccuracy, self.recall, self.precision],
                                        loss_weights=loss_weights, weighted_metrics=weighted_metrics,
                                        run_eagerly=run_eagerly, steps_per_execution=steps_per_execution,
                                        jit_compile=jit_compile, **kwargs)
        else:
            super(CNNMLP, self).compile(optimizer=self.optimizer, loss=loss, metrics=metrics + [self.categoricalAccuracy,
                self.recallClass0, self.recallClass1, self.recallClass2, self.recallClass3, self.recallClass4,
                self.precisionClass0, self.precisionClass1, self.precisionClass2, self.precisionClass3, self.precisionClass4],
                                        loss_weights=loss_weights, weighted_metrics=weighted_metrics,
                                        run_eagerly=run_eagerly, steps_per_execution=steps_per_execution,
                                        jit_compile=jit_compile, **kwargs)

        print(f"Compiled model with optimizer: {str(self.optimizer)}, "
              f"loss function: {self.loss}, "
              f"and metrics: {self.metrics}")



    def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose='auto',
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False):
        super().fit(x=x, y=y, verbose=verbose,
                    validation_split=validation_split, validation_data=validation_data, shuffle=shuffle,
                    class_weight=class_weight, sample_weight=sample_weight,
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                    validation_batch_size=validation_batch_size, validation_freq=validation_freq,
                    max_queue_size=max_queue_size,
                    workers=workers, use_multiprocessing=use_multiprocessing,
                    initial_epoch=self.epoch_counter.read_value().numpy(),
                    epochs=self.epoch_counter.read_value().numpy() + epochs,  # 1,
                    batch_size=None, callbacks=[self.csv_logger, self.lr_logger])
        self.epoch_counter.assign_add(epochs)  # 1)
        self.checkpoint.save(f"cnn_mlp_checkpoint/{self.epoch_counter.read_value().numpy()}")
        self.checkpoint_manager.save()



    @function(jit_compile = False) # Для вывода через tf.print нужно поставить False
    def train_step(self, data):
        x, y, weights = data
        x = self.AugmentData(x)

        if self.USE_BINARY_LABELS:
            y = tf.expand_dims(tf.cast(y[:, 0] > 0, dtype = tf.float32), axis = -1)


        if (tf.config.functions_run_eagerly()):
            xData = x.numpy()
            yData = y.numpy()

        with GradientTape() as tape:
            yPred = self(x, training=True)  # Forward pass

            # Compute our own loss
            loss = self.compiled_loss(y, yPred, regularization_losses = self.losses, sample_weight = weights)#loss(y, yPred, weights)

        if (tf.config.functions_run_eagerly()):
            yPredData = yPred.numpy()

        #t0 = time.time()

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        if (tf.config.functions_run_eagerly()):
            gradientsData = []
            for grads in gradients:
                gradientsData.append(grads.numpy())

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, yPred, weights)
        if not self.USE_BINARY_LABELS:
            yBinary = tf.expand_dims(tf.cast(y[:, 0] == 0, tf.float32), axis = -1)
            yPredBinary = tf.expand_dims(tf.cast(yPred[:, 0] < 0.5, tf.float32), axis = -1)
            self.binaryAccuracy.update_state(yBinary, yPredBinary, weights)
            self.precision.update_state(yBinary, yPredBinary, weights)
            self.recall.update_state(yBinary, yPredBinary, weights)

        return {m.name: m.result() for m in self.metrics}

    @function(jit_compile=False)  # Для вывода через tf.print нужно поставить False
    def test_step(self, data):
        x, y = data
        #currentLayer = self.AugmentData(currentLayer)

        if self.USE_BINARY_LABELS:
            y = tf.expand_dims(tf.cast(y[:, 0] > 0, dtype = tf.float32), axis = -1)

        if (tf.config.functions_run_eagerly()):
            xData = x.numpy()
            yData = y.numpy()

        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, regularization_losses = self.losses)  # loss(y, y_pred, weights)

        if (tf.config.functions_run_eagerly()):
            yPredData = y_pred.numpy()

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        if not self.USE_BINARY_LABELS:
            yBinary = tf.expand_dims(tf.cast(y[:, 0] == 0, tf.float32), axis = -1)
            yPredBinary = tf.expand_dims(tf.cast(y_pred[:, 0] < 0.5, tf.float32), axis = -1)
            self.binaryAccuracy.update_state(yBinary, yPredBinary)
            self.precision.update_state(yBinary, yPredBinary)
            self.recall.update_state(yBinary, yPredBinary)

        return {m.name: m.result() for m in self.metrics}


    @function(jit_compile=False)
    def predict_step(self, data):
        x = data

        if (tf.config.functions_run_eagerly()):
            xData = x.numpy()

        y = self(x, training=False)

        if (tf.config.functions_run_eagerly()):
            yData = y.numpy()

        return y


    def SetupImageNetWeights(self):
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


    @tf.function(jit_compile=False)
    def AugmentData(self, x):
        tempX = x

        #borderMask = tf.cast(tf.math.logical_and(x > 0.499, x < 0.501), dtype = tf.float32)

        # Аугментация 1: Зеркальное отражение по каждой оси
        tempX = tf.image.random_flip_left_right(tempX)
        tempX = tf.image.random_flip_up_down(tempX)

        # Аугментация 2: масштабирование (зум)
        tempX = self.zoomLayer(tempX, training = True)

        # Аугментация 3: сдвиг
        tempX = self.translateLayer(tempX, training = True)

        # Аугментация 4: модификация параметров изображения
        imageAugmentRatios = tf.random.uniform((4,), -1.0, 1.0)
        #tempX = tf.image.adjust_brightness(tempX, .3 * imageAugmentRatios[0])
        #tempX = tf.image.adjust_contrast(tempX, 1.0 + 0.18 * imageAugmentRatios[1])
        #tempX = tf.image.adjust_saturation(tempX, 1.0 + 0.18 * imageAugmentRatios[2])
        tempX = tf.image.adjust_hue(tempX, imageAugmentRatios[3])

        max = tf.reduce_max(tempX, axis = (1, 2, 3), keepdims = True)
        min = tf.reduce_min(tempX, axis = (1, 2, 3), keepdims = True)

        tempX = (tempX - min) / (max - min)

        #resultX = 0.5 * borderMask + x * (1 - borderMask)

        if (tf.config.functions_run_eagerly()):
            sourceXData = x.numpy()
            augmentedXData = tempX.numpy()
            #resultXData = resultX.numpy()

        #return resultX
        return tempX



    @tf.function(jit_compile=True)
    def PrepareImage(self, x):
        '''
        Подготовка изображения к подаче в нейронную сеть (увеличение контраста и т. д.)
        :param x: Исходное изображение сетчатки
        :return:
        '''
        indices = tf.constant(tf.cast(tf.stack(tf.meshgrid(tf.range(0, constants.IMAGE_SIZE[0]), tf.range(0, constants.IMAGE_SIZE[0]))), tf.float32) -
                              tf.constant([(constants.IMAGE_SIZE[0] - 1) / 2, (constants.IMAGE_SIZE[0] - 1) / 2], shape = (2, 1, 1)))
        indicesNorms = tf.constant(tf.norm(tf.cast(indices, tf.float32), axis = 0, ord = 2))

        x = x / 255.0
        sharpenedX = 0.5 + 4 * (x - tfa.image.gaussian_filter2d(x, 15, 10))

        nonZeros = tf.reduce_any(tf.greater_equal(x, tf.constant(0.05)), axis = -1)
        nonZeroRows = tf.reduce_any(nonZeros, axis = 2)
        nonZeroColumns = tf.reduce_any(nonZeros, axis = 1)

        if (tf.config.functions_run_eagerly()):
            nonZeroRowsData = nonZeroRows.numpy()
            nonZeroColumnsData = nonZeroColumns.numpy()

        nonZeroRowIndices = tf.cast(tf.where(nonZeroRows), tf.int32)
        nonZeroColumnIndices = tf.cast(tf.where(nonZeroColumns), tf.int32)

        rowMin = tf.reduce_min(nonZeroRowIndices)
        rowMax = tf.constant(constants.IMAGE_SIZE[0] - 1) - tf.reduce_max(nonZeroRowIndices)
        columnMin = tf.reduce_min(nonZeroColumnIndices)
        columnMax = tf.constant(constants.IMAGE_SIZE[1] - 1) - tf.reduce_max(nonZeroColumnIndices)

        radius = 0.9 * 0.5 * (tf.constant(constants.IMAGE_SIZE[0], tf.float32) - tf.maximum(tf.cast(rowMin + rowMax, tf.float32), tf.cast(columnMin + columnMax, tf.float32)))
        normalizedRadius = radius / constants.IMAGE_SIZE[0]
        radialMask = tf.expand_dims(tf.cast(indicesNorms < tf.cast(radius, tf.float32), tf.float32), axis = -1)

        maskedSharpenedX = sharpenedX * radialMask + 0.5 * (1 - radialMask)
        croppedMaskedSharpenedX = tf.image.crop_and_resize(maskedSharpenedX, [[0.5 - normalizedRadius, 0.5 - normalizedRadius, 0.5 + normalizedRadius, 0.5 + normalizedRadius]], [0], constants.IMAGE_SIZE)

        if (tf.config.functions_run_eagerly()):
            imgData = tf.squeeze(x).numpy()

            maskData = radialMask.numpy()
            maskedImgData = imgData * maskData

            resultData = sharpenedX.numpy()
            maskedResultData = resultData * maskData + 0.5 * (1 - maskData)

            normalizedRadius = radius / constants.IMAGE_SIZE[0]
            croppedMaskedResultData = tf.image.crop_and_resize(maskedResultData, [[0.5 - normalizedRadius, 0.5 - normalizedRadius, 0.5 + normalizedRadius, 0.5 + normalizedRadius]], [0], constants.IMAGE_SIZE).numpy()

        return croppedMaskedSharpenedX


def GetMinMaxNonZeros(x):
    nonZeroIndices = tf.cast(tf.where(x), tf.int32)
    return [tf.slice(nonZeroIndices, tf.cast([0, 0], tf.int64), tf.cast([1, 1], tf.int64)), tf.constant(constants.IMAGE_SIZE[0] - 1) - tf.slice(nonZeroIndices, tf.cast([tf.shape(nonZeroIndices)[0] - 1, 0], tf.int64), tf.cast([1, 1], tf.int64))]
