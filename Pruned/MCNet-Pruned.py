# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
import os,random
import time
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt
from .. import mltools,rmldataset2016
from .. import rmlmodels
import csv
from keras.models import Model, Layer
from keras.layers import Input
import tensorflow as tf
import keras
from keras.layers import Dropout, Reshape, BatchNormalization, MultiHeadAttention, GlobalAveragePooling1D
from  keras import backend as K
import tensorflow_model_optimization as tfmot
import tempfile
from tensorflow.keras.callbacks import Callback
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from prunable_layers import PruneBidirectional
class SumLayer(Layer):
    def __init__(self, **kwargs):
        super(SumLayer, self).__init__(**kwargs)

    def call(self, inputs):
        summed_output = tf.reduce_sum(inputs, axis=1)
        return summed_output

class GaussianNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, snr, **kwargs):
        super(GaussianNoiseLayer, self).__init__(**kwargs)
        self.snr = snr

    def call(self, x, training=None):
        noise_shape = tf.shape(x)
        mean = 0.0
        std_dev = 1.0
        random_noise = tf.random.normal(noise_shape, mean=mean, stddev=std_dev)
        random_noise_energy = tf.reduce_sum(tf.abs(random_noise) ** 2)
        signal_energy = tf.reduce_sum(tf.square(x))
        signal_energy_expected = random_noise_energy * (10 ** (self.snr / 10.0))
        normalized_x = x * tf.sqrt(signal_energy_expected) / tf.sqrt(signal_energy)
        total_vector = normalized_x + random_noise
        return total_vector
    def get_config(self):
        config = super(GaussianNoiseLayer, self).get_config()
        config.update({"snr": self.snr})
        return config

(mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = \
    rmldataset2016.load_data()
in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods
print(classes)

def print_model_sparsity(pruned_model):
    def _get_sparsity(weights):
        return 1.0 - np.count_nonzero(weights) / float(weights.size)

    print("Model Sparsity Summary ({})".format(pruned_model.name))
    print("--")
    for layer in pruned_model.layers:
        if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            prunable_weights = layer.layer.get_prunable_weights()
            if prunable_weights:
                print("{}: {}".format(
                    layer.name, ", ".join([
                        "({}, {})".format(weight.name,
                                          str(_get_sparsity(K.get_value(weight))))
                        for weight in prunable_weights
                    ])))
    print("\n")
# Set up some params
nb_epoch = 300    # number of epochs to train on
batch_size = 200  # training batch size
print(batch_size)
model = rmlmodels.C_AMCModel.PrunedModel(weights=None, input_shape=[512, 2], classes=11)
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='Adam')
model.summary()
filepath = 'weights/weights1.h5'
model.load_weights(filepath)
def apply_pruning_to_dense(layer):
    if isinstance(layer, PruneBidirectional):
        return prune.prune_low_magnitude(layer, pruning_schedule.PolynomialDecay(
            initial_sparsity=0.1, final_sparsity=0.8, begin_step=0, end_step=6600))
    if isinstance(layer, keras.layers.Dense):
        if layer.name == 'xd':
            return prune.prune_low_magnitude(layer, pruning_schedule.PolynomialDecay(
                initial_sparsity=0.1, final_sparsity=0.8, begin_step=0, end_step=6600))
    elif isinstance(layer, keras.layers.MultiHeadAttention):
        return prune.prune_low_magnitude(layer, pruning_schedule.PolynomialDecay(
            initial_sparsity=0.1, final_sparsity=0.8, begin_step=0, end_step=6600))
    return layer

pruned_model = tf.keras.models.clone_model(
    model,
    clone_function=apply_pruning_to_dense,
)
print_model_sparsity(pruned_model)
pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

pruned_model.fit(X_train, Y_train,batch_size=batch_size,
                    epochs=15,
                    verbose=2,
                    callbacks = [pruning_callbacks.UpdatePruningStep()],
                    validation_data=(X_val, Y_val))
print_model_sparsity(pruned_model)
pruned_model.summary()
model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
print_model_sparsity(model_for_export)
pruned_keras_file = 'pruned_model.h5'
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
model = model_for_export
model.summary()

input_layer0 = Input(shape=(512, 2), name='input')
output_layer0 = model.get_layer('Conv1')(input_layer0)
output_layer0 = Dropout(0.5)(output_layer0)
output_layer0 = model.get_layer('Conv2')(output_layer0)
output_layer0 = Dropout(0.5)(output_layer0)
output_layer0 = SumLayer()(output_layer0)
output_layer0 = BatchNormalization()(output_layer0)
output_layer0 = model.get_layer('dense1')(output_layer0)
output_layer0 = BatchNormalization()(output_layer0)

compress_model = Model(inputs=input_layer0, outputs=output_layer0)

for i in range(len(compress_model.layers)):
    compress_model.layers[i].set_weights(model.layers[i].get_weights())
compress_model.save_weights('weights/compress_weights.h5')


input_layer = Input(shape=(64,1), name='input1')
output_layer = model.get_layer('lstm3')(input_layer)
output_layer = Dropout(0.5)(output_layer)
output_layer = model.get_layer('lstm4')(output_layer)
output_layer = Reshape((1,128))(output_layer)
multi_head_attention = MultiHeadAttention(num_heads=8, key_dim=128)
output_layer = multi_head_attention(query=output_layer, key=output_layer, value=output_layer)
global_avg_pool = GlobalAveragePooling1D()
output_layer = global_avg_pool(output_layer)
output_layer = model.get_layer('dense4')(output_layer)
output_layer = BatchNormalization()(output_layer)
output_layer = Dropout(0.5)(output_layer)
output_layer = model.get_layer('xd')(output_layer)
cla_model = Model(inputs=input_layer, outputs=output_layer)

for i in range(1, len(cla_model.layers)):
    cla_model.layers[i].set_weights(model.layers[i+len(model.layers)-11].get_weights())
cla_model.save_weights('weights/cla_weights.h5')

def predict(compress_model, cla_model):


    test_Y_C = compress_model.predict(X_test, batch_size=batch_size)
    gaussian_noise_layer = GaussianNoiseLayer(10)  
    test_Y_C = gaussian_noise_layer(test_Y_C)
    test_Y_C = Reshape((64,1))(test_Y_C)
    test_Y_hat = cla_model.predict(test_Y_C, batch_size=batch_size)

    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    # Plot confusion matrix
    mltools.plot_confusion_matrix(confnorm,
                                  labels=['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', '4-PAM', '16-QAM',
                                          '64-QAM', 'QPSK', 'WBFM'], save_filename='figure/lstm3_total_confusion.png')

    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    i = 0
    for snr in snrs:

        test_SNRs = [lbl[x][1] for x in test_idx]

        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        test_Y_i_C = compress_model.predict(test_X_i, batch_size=200)

        gaussian_noise_layer = GaussianNoiseLayer(10)
        test_Y_i_C = gaussian_noise_layer(test_Y_i_C)
        test_Y_i_C = Reshape((64,1))(test_Y_i_C)
        test_Y_i_hat = cla_model.predict(test_Y_i_C)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)
        acc[snr] = 1.0 * cor / (cor + ncor)
        result = cor / (cor + ncor)
        with open('acc111.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([result])
        mltools.plot_confusion_matrix(confnorm_i,
                                      labels=['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', '4-PAM', '16-QAM',
                                              '64-QAM', 'QPSK', 'WBFM'], title="Confusion Matrix",
                                      save_filename="figure/Confusion(SNR=%d)(ACC=%2f).png" % (snr, 100.0 * acc[snr]))

        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3)
        i = i + 1

    # plot acc of each mod in one picture
    dis_num = 11
    for g in range(int(np.ceil(acc_mod_snr.shape[0] / dis_num))):
        assert (0 <= dis_num <= acc_mod_snr.shape[0])
        beg_index = g * dis_num
        end_index = np.min([(g + 1) * dis_num, acc_mod_snr.shape[0]])

        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy for Each Mod")

        markers = ['o', 's', '^', 'd', '*', '+', 'x', 'p', 'h', '.', ',']
        for i in range(beg_index, end_index):
            plt.plot(snrs, acc_mod_snr[i], label=classes[i], marker=markers[i % len(markers)], markersize=8)
            for x, y in zip(snrs, acc_mod_snr[i]):
                plt.text(x, y, y, ha='center', va='bottom', fontsize=8)

        plt.legend()
        plt.grid()
        plt.savefig('figure/acc_with_mod_{}.png'.format(g + 1))
        plt.close()
    np.save('acc_mod_snr.npy', acc_mod_snr)
    acc_mod_snr_mean = np.mean(acc_mod_snr, axis=0)
    np.savetxt('NLSTML16_accuracy.txt', acc_mod_snr_mean)
    plt.figure(figsize=(12, 10))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Average Classification Accuracy1")
    plt.plot(snrs, acc_mod_snr_mean)
    plt.grid()
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.savefig('figure/Average Classification Accuracy.png')
    plt.close()

predict(compress_model, cla_model)