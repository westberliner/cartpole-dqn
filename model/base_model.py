import os
import numpy as np
import tensorflow as tf
import datetime

class BaseModel:

    model_path = 'trained_models'
    logging = True
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = []

    def __init__(self, input_shape):
        self.path_to_model = '{}/{}.h5'.format(self.model_path, self.model_name)
        self.input_shape = input_shape
        if (self.load() != True):
            self.set_model()
        
        if self.logging:
            self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1))
    
    def set_model(self):
        pass

    def predict(self, state):
        state = np.array([state])
        prediction = self.model.predict(state)
        return np.argmax(prediction[0])

    def translate_to_one_hot_target(self, action):
        one_hot_action = []
        for i in range(2):
            if action == i:
                one_hot_action.append(1)
            else:
                one_hot_action.append(10)
        return np.array(one_hot_action)

    def fit(self, x_val, y_val, epochs=1, verbose=1):
        self.model.fit(x_val, y_val, epochs=epochs, verbose=verbose, callbacks=self.callbacks)

    def save(self):
        self.model.save(self.path_to_model)

    def load(self):
        if os.path.isfile(self.path_to_model) != True:
            print("load failed")
            return False
        self.model = tf.keras.models.load_model(self.path_to_model)
        print("loaded")
        return True
        

