from tensorflow.keras.models import Model, Sequential
#from keras.layers import Dense
#from tensorflow.keras.initializers import Zeros, RandomNormal, HeNormal, HeUniform
#from keras.regularizers import L2
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.metrics import Mean, CategoricalAccuracy, Recall, Precision

import constants
from constants import RANDOM_SEED, BATCH_SIZE
from tensorflow import train, Variable, int32, function, GradientTape
from tools import FitLogCallback
import time

from tensorflow.keras.optimizers import Adam
#from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras.optimizers import SGD

import tensorflow as tf
import keras as keras

LEARNING_RATE = 2.0e-03
MOMENTUM = 0.9
EPSILON = 1.0e-03
DECAY = 1.0e-02
AMSGRAD = True

class CNNMLP(Model):
    def __init__(self, *args, **kwargs):
        super(CNNMLP, self).__init__(*args, **kwargs)

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
        self.optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=MOMENTUM, epsilon = EPSILON, amsgrad = AMSGRAD, decay = DECAY)#SGD(learning_rate = LEARNING_RATE, momentum = MOMENTUM, nesterov = True, decay = DECAY)#
        self.epoch_counter = Variable(initial_value=0, dtype=int32, trainable=False)
        self.checkpoint = train.Checkpoint(model = self, optimizer = self.optimizer, epoch_counter=self.epoch_counter)
        self.checkpoint_manager = train.CheckpointManager(checkpoint = self.checkpoint, directory = "cnn_mlp_checkpoint", max_to_keep = 10)
        print(f"CNNMLP last checkpoint: {self.checkpoint_manager.latest_checkpoint}")
        #print(f"CNNMLP last checkpoint path: {self.checkpoint_manager.restore_or_initialize()}")
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        #if (self.checkpoint_manager.latest_checkpoint is None):
            #print("Assigned bias (0 for class #0 and 1 for others)")
            #lastLayersWeights = self.get_layer("fc_predict").get_weights()
            #lastLayersWeights[1] = Variable([0.0, 1.0, 1.0, 1.0, 1.0], dtype = tf.float32, trainable = True)
            #self.getlayer("fc_predict").set_weights(lastLayersWeights)

        self.logger = FitLogCallback.FitCallback()
        self.csv_logger = CSVLogger(filename=self.checkpoint_manager.directory + "_training_log.csv", append=True)
        self.lr_logger = ReduceLROnPlateau(monitor = "loss", factor = 0.5, patience = 10, verbose = 2,
                                           min_delta = 0.001, cooldown = 2, min_lr = 1.0e-06)
        self.loss_tracker = Mean(name = "loss")
        self.accuracy_metric = CategoricalAccuracy()
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

        print(f"Initialized CNNMLP for epoch #{self.epoch_counter.read_value().numpy()}")


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.accuracy_metric]#,
                #self.recallClass0, self.recallClass1, self.recallClass2, self.recallClass3, self.recallClass4,
                #self.precisionClass0, self.precisionClass1, self.precisionClass2, self.precisionClass3, self.precisionClass4]


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
        #self.optimizer.learning_rate.assign(LEARNING_RATE)
        #self.optimizer.momentum.assign(MOMENTUM)
        #self.optimizer.epsilon = EPSILON
        #self.optimizer.amsgrad = AMSGRAD
        super(CNNMLP, self).compile(optimizer = self.optimizer, loss = loss, metrics = metrics + [self.accuracy_metric],#, self.recallClass0, self.recallClass1, self.recallClass2, self.recallClass3, self.recallClass4,
                #self.precisionClass0, self.precisionClass1, self.precisionClass2, self.precisionClass3, self.precisionClass4],
                loss_weights = loss_weights, weighted_metrics = weighted_metrics, run_eagerly = run_eagerly, steps_per_execution = steps_per_execution, jit_compile = jit_compile, **kwargs)
        print(f"Compiled model with optimizer: {str(self.optimizer)}, "
              f"loss function: {self.loss}, "
              f"and metrics: {self.metrics}")
        #self.optimizer.learning_rate = 3e-02
        #self.optimizer.learning_rate.assign(LEARNING_RATE)
        #self.optimizer.momentum.assign(MOMENTUM)
        #self.optimizer.epsilon = EPSILON
        #self.optimizer.amsgrad = AMSGRAD



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

        for i in range(1):#epochs):
            trainData = \
                x.\
                    shuffle(200, reshuffle_each_iteration = True).\
                    batch(BATCH_SIZE, drop_remainder = True).\
                    prefetch(tf.data.AUTOTUNE)
            testData = \
                validation_data.\
                    shuffle(200, reshuffle_each_iteration = True).\
                    batch(BATCH_SIZE, drop_remainder = True).\
                    prefetch(tf.data.AUTOTUNE)

            super().fit(x = trainData, y = y, verbose = verbose,
                               validation_split = validation_split, validation_data = testData, shuffle = shuffle,
                               class_weight = class_weight, sample_weight = sample_weight,
                               steps_per_epoch = steps_per_epoch, validation_steps = validation_steps,
                               validation_batch_size = validation_batch_size, validation_freq = validation_freq, max_queue_size = max_queue_size,
                               workers = workers, use_multiprocessing = use_multiprocessing, initial_epoch = self.epoch_counter.read_value().numpy(), epochs = self.epoch_counter.read_value().numpy() + epochs,#1,
                               batch_size = None, callbacks = [self.logger, self.csv_logger, self.lr_logger])
            #x = x.shuffle(buffer_size = 200)
            #validation_data = validation_data.shuffle(buffer_size = 200)
            self.epoch_counter.assign_add(epochs)#1)

            #result = self.predict(x.batch(BATCH_SIZE).take(1))
            #print(result)


        self.checkpoint.save(f"cnn_mlp_checkpoint/{self.epoch_counter.read_value().numpy()}")
        self.checkpoint_manager.save()



    @function(jit_compile = False) # Для вывода через tf.print нужно поставить False
    def train_step(self, data):
        #tf.print(":)")
        x, y, weights = data

        if (tf.config.functions_run_eagerly()):
            xData = x.numpy()
            yData = y.numpy()

        t0 = time.time()
        with GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            #f.print("Network respond:\n", y_pred)
            # Compute our own loss
            loss = self.compiled_loss(y, y_pred, sample_weight = weights, regularization_losses = self.losses)#loss(y, y_pred, weights)

        if (tf.config.functions_run_eagerly()):
            yPredData = y_pred.numpy()

        print(f"Network time: {time.time() - t0}")
        t0 = time.time()

        # Compute gradients
        #trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, self.trainable_weights)

        print(f"Gradient time: {time.time() - t0}")
        t0 = time.time()

        if (tf.config.functions_run_eagerly()):
            gradientsData = []
            for grads in gradients:
                gradientsData.append(grads.numpy())

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        print(f"Apply weights time: {time.time() - t0}")
        t0 = time.time()

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred, weights)
        '''
        self.recallClass0.update_state(y, y_pred)
        self.recallClass1.update_state(y, y_pred)
        self.recallClass2.update_state(y, y_pred)
        self.recallClass3.update_state(y, y_pred)
        self.recallClass4.update_state(y, y_pred)
        self.precisionClass0.update_state(y, y_pred)
        self.precisionClass1.update_state(y, y_pred)
        self.precisionClass2.update_state(y, y_pred)
        self.precisionClass3.update_state(y, y_pred)
        self.precisionClass4.update_state(y, y_pred)
        '''

        return {m.name: m.result() for m in self.metrics}

    @function(jit_compile=False)  # Для вывода через tf.print нужно поставить False
    def test_step(self, data):
        # tf.print(":)")
        x, y = data

        if (tf.config.functions_run_eagerly()):
            xData = x.numpy()
            yData = y.numpy()

        t0 = time.time()
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, regularization_losses = self.losses)  # loss(y, y_pred, weights)

        if (tf.config.functions_run_eagerly()):
            yPredData = y_pred.numpy()

        print(f"Network time: {time.time() - t0}")

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        '''
        self.recallClass0.update_state(y, y_pred)
        self.recallClass1.update_state(y, y_pred)
        self.recallClass2.update_state(y, y_pred)
        self.recallClass3.update_state(y, y_pred)
        self.recallClass4.update_state(y, y_pred)
        self.precisionClass0.update_state(y, y_pred)
        self.precisionClass1.update_state(y, y_pred)
        self.precisionClass2.update_state(y, y_pred)
        self.precisionClass3.update_state(y, y_pred)
        self.precisionClass4.update_state(y, y_pred)
        '''

        #tf.print("TEST RESULT")
        #tf.print("y_true:\n", y)
        #tf.print("y_pred:\n", y_pred)
        #tf.print("-------")

        return {m.name: m.result() for m in self.metrics}


    @function(jit_compile = False)
    def CallNetwork(self, x, training):
        return self(x, training = training)

