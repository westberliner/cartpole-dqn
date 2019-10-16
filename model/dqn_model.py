import tensorflow as tf
from .base_model import BaseModel

class DQNModel(BaseModel):
    
    model_name = 'dqn_cartpole'
    
    def set_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation='relu', input_shape=self.input_shape),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear')
        ])
        self.model.compile(loss='mse', optimizer=tf.optimizers.Adam(lr=0.001), metrics=['accuracy'])
