# prunable_layers.py
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer

class PruneBidirectional(Bidirectional, prunable_layer.PrunableLayer):
    def get_prunable_weights(self):

        forward_weights = self.forward_layer.trainable_weights

        backward_weights = self.backward_layer.trainable_weights
   
        return forward_weights + backward_weights