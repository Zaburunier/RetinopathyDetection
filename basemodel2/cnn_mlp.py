from keras.models import Model, Sequential
from keras.layers import Dense
from keras.initializers.initializers_v2 import Zeros, RandomNormal
from keras.regularizers import L2
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from keras.metrics import Mean, CategoricalAccuracy

import constants
from constants import RANDOM_SEED, BATCH_SIZE
from tensorflow import train, Variable, int32, function, GradientTape
from tools import FitLogCallback
import time

from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.gradient_descent import SGD

import tensorflow as tf

class CNNMLP(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        '''
        cnnNetwork = BuildKerasCNN()
        self.network = Sequential(name="cnn_mlp")
        self.network.add(cnnNetwork)
        
        mlp_predictor = Dense(5,
                              name="fc_predict",
                              activation=None,
                              kernel_initializer=RandomNormal(mean=0.0, stddev=1e-02, seed=RANDOM_SEED),
                              bias_initializer=Zeros(),
                              kernel_regularizer=L2(l2 = 1e-03))
        self.network.add(mlp_predictor)
        '''
        self.optimizer = Adam(learning_rate=3.0e-04, beta_1=0.9, epsilon = 1e-03)#, amsgrad = True)#SGD(learning_rate = 1.0e-04, momentum = 0.1)

        self.epoch_counter = Variable(initial_value=0, dtype=int32, trainable=False)
        self.checkpoint = train.Checkpoint(model = self, optimizer = self.optimizer, epoch_counter=self.epoch_counter)
        self.checkpoint_manager = train.CheckpointManager(checkpoint = self.checkpoint, directory = "cnn_mlp_checkpoint", max_to_keep = 10)
        print(f"CNNMLP last checkpoint: {self.checkpoint_manager.latest_checkpoint}")
        #print(f"CNNMLP last checkpoint path: {self.checkpoint_manager.restore_or_initialize()}")
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        if (self.checkpoint_manager.latest_checkpoint is None):
            print("Assigned bias (0 for class #0 and 1 for others)")
            lastLayersWeights = self.get_layer("fc_predict").get_weights()
            lastLayersWeights[1] = Variable([0.0, 1.0, 1.0, 1.0, 1.0], dtype = tf.float32, trainable = True)
            self.get_layer("fc_predict").set_weights(lastLayersWeights)

        self.logger = FitLogCallback.FitCallback()
        self.csv_logger = CSVLogger(filename=self.checkpoint_manager.directory + "_training_log.csv", append=True)
        self.loss_tracker = Mean(name = "loss")
        self.accuracy_metric = CategoricalAccuracy()

        print(f"Initialized CNNMLP for epoch #{self.epoch_counter.read_value().numpy()}")


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.accuracy_metric]


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
        super().compile(optimizer = self.optimizer, loss = loss, metrics = metrics + [self.accuracy_metric], loss_weights = loss_weights, weighted_metrics = weighted_metrics, run_eagerly = run_eagerly, steps_per_execution = steps_per_execution, jit_compile = jit_compile, **kwargs)
        print(f"Compiled model with optimizer: {self.optimizer}, "
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
        #x = x.shuffle(buffer_size=200, seed = constants.RANDOM_SEED)
        #validation_data = validation_data.shuffle(buffer_size=200, seed = constants.RANDOM_SEED)

        for i in range(epochs):
            trainData = x.batch(BATCH_SIZE, drop_remainder = True).prefetch(tf.data.AUTOTUNE)
            testData = validation_data.batch(BATCH_SIZE, drop_remainder = True).prefetch(tf.data.AUTOTUNE)

            super().fit(x = trainData, y = y, verbose = verbose,
                               validation_split = validation_split, validation_data = testData, shuffle = shuffle,
                               class_weight = class_weight, sample_weight = sample_weight,
                               steps_per_epoch = steps_per_epoch, validation_steps = validation_steps,
                               validation_batch_size = validation_batch_size, validation_freq = validation_freq, max_queue_size = max_queue_size,
                               workers = workers, use_multiprocessing = use_multiprocessing, initial_epoch = self.epoch_counter.read_value().numpy(), epochs = self.epoch_counter.read_value().numpy() + 1,
                               batch_size = None, callbacks = [self.logger, self.csv_logger])
            x = x.shuffle(buffer_size = 200)
            validation_data = validation_data.shuffle(buffer_size = 200)
            self.epoch_counter.assign_add(1)

            #result = self.predict(x.batch(BATCH_SIZE).take(1))
            #print(result)


        self.checkpoint.save(f"cnn_mlp_checkpoint/{self.epoch_counter.read_value().numpy()}")
        self.checkpoint_manager.save()



    @function(jit_compile = True) # Для вывода через tf.print нужно поставить False
    def train_step(self, data):
        #tf.print(":)")
        x, y, weights = data

        if (tf.config.functions_run_eagerly()):
            xData = x.numpy()

        t0 = time.time()
        with GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            #f.print("Network respond:\n", y_pred)
            # Compute our own loss
            loss = 10.0 * self.compiled_loss(y, y_pred, weights)

        print(f"Network time: {time.time() - t0}")
        t0 = time.time()

        # Compute gradients
        #trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, self.trainable_weights)

        print(f"Gradient time: {time.time() - t0}")
        t0 = time.time()

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        print(f"Apply weights time: {time.time() - t0}")
        t0 = time.time()

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_metric.update_state(y, y_pred, weights)

        return {m.name: m.result() for m in self.metrics}

