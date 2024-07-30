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
from keras.layers import Input, Bidirectional
import tensorflow as tf
import keras
from keras.layers import Dropout, Reshape, BatchNormalization, MultiHeadAttention, GlobalAveragePooling1D
from  keras import backend as K
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
classes = mods
print(classes)
# Set up some params
nb_epoch = 300    # number of epochs to train on
batch_size = 200  # training batch size

pruned_keras_file = 'pruned_model.h5'
model = tf.keras.models.load_model(pruned_keras_file,custom_objects={'SumLayer': SumLayer, 'GaussianNoiseLayer':GaussianNoiseLayer,'PruneBidirectional':PruneBidirectional
})
model.summary()

i = 0
unquan_weights = []
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv1D):
        unquan_weights.append(layer.get_weights())
        i += 1
print(i)

k=0
unquan_weights1= []
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
         unquan_weights1.append(layer.get_weights())
         k += 1
print(k)

quan_weights = unquan_weights
quan_weights1 = unquan_weights1

# Quantization bit number
bits = 8

# Quantize the weights of the Conv1D layer
for p in range(i):
    #Get weights and biases
    weights = quan_weights[p]
    weights_kernel, weights_bias = weights

    # Process weight matrix
    weights_kernel_flat = np.reshape(weights_kernel, (-1,))
    nonzero_indices = weights_kernel_flat != 0
    nonzero_weights = weights_kernel_flat[nonzero_indices]
    weights_kernel_max = np.max(nonzero_weights)
    weights_kernel_min = np.min(nonzero_weights)

    # Normalize non-zero weights and quantize
    normalized_weights = (nonzero_weights - weights_kernel_min) / (weights_kernel_max - weights_kernel_min)
    quantized_weights = np.round(normalized_weights * (2 ** bits - 1))
    quantized_weights = quantized_weights / (2 ** bits - 1) * (weights_kernel_max - weights_kernel_min) + weights_kernel_min

    # Put quantized weights back to the original array
    weights_kernel_flat[nonzero_indices] = quantized_weights
    weights_kernel_quantized = np.reshape(weights_kernel_flat, weights_kernel.shape)

    # Process bias
    weights_bias_flat = np.reshape(weights_bias, (-1,))
    nonzero_indices_bias = weights_bias_flat != 0
    nonzero_bias = weights_bias_flat[nonzero_indices_bias]
    bias_max = np.max(nonzero_bias)
    bias_min = np.min(nonzero_bias)

    # Normalize non-zero bias and quantize
    normalized_bias = (nonzero_bias - bias_min) / (bias_max - bias_min)
    quantized_bias = np.round(normalized_bias * (2 ** bits - 1))
    quantized_bias = quantized_bias / (2 ** bits - 1) * (bias_max - bias_min) + bias_min

    # Put quantized bias back to the original array
    weights_bias_flat[nonzero_indices_bias] = quantized_bias
    weights_bias_quantized = np.reshape(weights_bias_flat, weights_bias.shape)

    # Update quantized weights
    quan_weights[p] = [weights_kernel_quantized, weights_bias_quantized]

# Quantize the weights of the Dense layer
for p in range(k):
    # Get weights and biases
    weights = quan_weights1[p]
    weights_kernel, weights_bias = weights

    # Process weight matrix
    weights_kernel_vec = np.reshape(weights_kernel, (np.size(weights_kernel),))
    weights_kernel_nozero = np.delete(weights_kernel_vec, np.argwhere(weights_kernel_vec == 0))
    weights_kernel_max = np.max(weights_kernel_nozero)
    weights_kernel_min = np.min(weights_kernel_nozero)
    nonzero_index_kernel = np.transpose(np.array(np.nonzero(weights_kernel)))
    len_index_kernel = len(nonzero_index_kernel)
    for m in range(len_index_kernel):
        weights_kernel[nonzero_index_kernel[m][0], nonzero_index_kernel[m][1]] = np.round(
            (weights_kernel[nonzero_index_kernel[m][0], nonzero_index_kernel[m][1]] - weights_kernel_min) /
            (weights_kernel_max - weights_kernel_min) * (2 ** bits - 1)) * (
                                                                                         weights_kernel_max - weights_kernel_min) / (
                                                                                         2 ** bits - 1) + weights_kernel_min
    # Process bias
    weights_bias_vec = np.reshape(weights_bias, (np.size(weights_bias),))
    weights_bias_nozero = np.delete(weights_bias_vec, np.argwhere(weights_bias_vec == 0))
    weights_bias_max = np.max(weights_bias_nozero)
    weights_bias_min = np.min(weights_bias_nozero)
    nonzero_index_bias = np.transpose(np.array(np.nonzero(weights_bias)))
    len_index_bias = len(nonzero_index_bias)
    for m in range(len_index_bias):
        weights_bias[nonzero_index_bias[m][0]] = np.round(
            (weights_bias[nonzero_index_bias[m][0]] - weights_bias_min) / (weights_bias_max - weights_bias_min) * (
                    2 ** bits - 1)) * (weights_bias_max - weights_bias_min) / (2 ** bits - 1) + weights_bias_min

    # Update quantized weights
    quan_weights1[p] = [weights_kernel, weights_bias]


j = 0
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv1D):
        layer.set_weights(quan_weights[j])
        j += 1
print(j)
n=0
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        layer.set_weights(quan_weights1[n])
        n += 1
print(n)


def quantize_bidirectional_layer(layer, bits):
    quantized_weights = []
    for sub_layer in [layer.forward_layer, layer.backward_layer]:

        weights = sub_layer.get_weights()
        quantized_sub_weights = []

        for weight in weights:

            quantized_weight = quantize_weights_b(weight, bits)
            quantized_sub_weights.append(quantized_weight)
        quantized_weights.append(quantized_sub_weights)

    layer.forward_layer.set_weights(quantized_weights[0])
    layer.backward_layer.set_weights(quantized_weights[1])
def quantize_weights_b(weights, bits):

    weights_nonzero = weights[weights != 0]
    min_val = np.min(weights_nonzero)
    max_val = np.max(weights_nonzero)

    normalized_weights = (weights - min_val) / (max_val - min_val)
    quantized_weights = np.round(normalized_weights * (2 ** bits - 1))
    quantized_weights = quantized_weights / (2 ** bits - 1) * (max_val - min_val) + min_val

    return quantized_weights

def quantize_multihead_attention_layer(layer, bits):

    weights = layer.get_weights()
    quantized_weights = []

    for weight in weights:
        quantized_weight = quantize_weights(weight, bits)
        quantized_weights.append(quantized_weight)

    layer.set_weights(quantized_weights)

def quantize_weights(weights, bits):
    weights_nonzero = weights[weights != 0]
    if weights_nonzero.size == 0:
        return weights

    min_val = np.min(weights_nonzero)
    max_val = np.max(weights_nonzero)

    normalized_weights = (weights - min_val) / (max_val - min_val)
    quantized_weights = np.round(normalized_weights * (2 ** bits - 1))
    quantized_weights = quantized_weights / (2 ** bits - 1) * (max_val - min_val) + min_val

    return quantized_weights

def quantize_model(model, bits):
    for layer in model.layers:

        if isinstance(layer, PruneBidirectional):
            quantize_bidirectional_layer(layer, bits)
        if isinstance(layer, MultiHeadAttention):
            quantize_multihead_attention_layer(layer, bits)
quantize_model(model, 8)
model.save('quan_model.h5')

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

    start_time = time.time()
    test_Y_C = compress_model.predict(X_test, batch_size=batch_size)
    end_time = time.time()
    print(end_time - start_time)
    gaussian_noise_layer = GaussianNoiseLayer(10)
    test_Y_C = gaussian_noise_layer(test_Y_C)
    test_Y_C = Reshape((64,1))(test_Y_C)
    test_Y_hat = cla_model.predict(test_Y_C, batch_size=batch_size)
    end_time1 = time.time()
    print(end_time1 - start_time)

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
        # estimate classes
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
    np.savetxt('C-AMC Pruned and Quantized accuracy.txt', acc_mod_snr_mean)
    plt.figure(figsize=(12, 10))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Average Classification Accuracy")
    plt.plot(snrs, acc_mod_snr_mean)
    plt.grid()
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.savefig('figure/Average Classification Accuracy.png')
    plt.close()

predict(compress_model, cla_model)