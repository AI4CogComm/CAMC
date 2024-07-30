
import pickle
import numpy as np
from numpy import linalg as la 

maxlen=512
def l2_normalize(x, axis=-1):
    y = np.max(np.sum(x ** 2, axis, keepdims=True), axis, keepdims=True)
    return x / np.sqrt(y)

def norm_pad_zeros(X_train,nsamples):
    print ("Pad:", X_train.shape)
    for i in range(X_train.shape[0]):
        X_train[i,:,0] = X_train[i,:,0]/la.norm(X_train[i,:,0],2)
    return X_train
def to_amp_phase(X_train, X_val, X_test, nsamples):
    X_train_cmplx = X_train[:, :,0] + 1j * X_train[:, :,1]
    X_val_cmplx = X_val[:, :,0] + 1j * X_val[:, :,1]
    X_test_cmplx = X_test[:, :,0] + 1j * X_test[:, :,1]

    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:, :,1], X_train[:, :,0]) / np.pi

    X_train_amp = np.reshape(X_train_amp, (-1, 1, nsamples))
    X_train_ang = np.reshape(X_train_ang, (-1, 1, nsamples))

    X_train = np.concatenate((X_train_amp, X_train_ang), axis=1)
    X_train = np.transpose(np.array(X_train), (0, 2, 1))

    X_val_amp = np.abs(X_val_cmplx)
    X_val_ang = np.arctan2(X_val[:, :,1], X_val[:, :,0]) / np.pi

    X_val_amp = np.reshape(X_val_amp, (-1, 1, nsamples))
    X_val_ang = np.reshape(X_val_ang, (-1, 1, nsamples))

    X_val = np.concatenate((X_val_amp, X_val_ang), axis=1)
    X_val = np.transpose(np.array(X_val), (0, 2, 1))

    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:, :,1], X_test[:, :,0]) / np.pi

    X_test_amp = np.reshape(X_test_amp, (-1, 1, nsamples))
    X_test_ang = np.reshape(X_test_ang, (-1, 1, nsamples))

    X_test = np.concatenate((X_test_amp, X_test_ang), axis=1)
    X_test = np.transpose(np.array(X_test), (0, 2, 1))
    return (X_train, X_val, X_test)

def downsample(signal, factor=2):
    length = signal.shape[1] - (signal.shape[1] % factor)
    signal = signal[:, :length]
    return signal.reshape(signal.shape[0], length // factor, factor).mean(axis=2)
def load_data():
    X_train = np.load('D:/HECHAOWEI/TFProject4\data2/train_data.npy')
    X_val = np.load('D:\HECHAOWEI\TFProject4\data2/val_data.npy')
    X_test = np.load('D:\HECHAOWEI\TFProject4\data2/test_data.npy')

    # Simple downsampling
    X_train = X_train[:, ::2, :]
    X_val = X_val[:, ::2, :]
    X_test = X_test[:, ::2, :]

    Y_train = np.load('D:\HECHAOWEI\TFProject4\data2/train_labels_one_hot.npy')
    Y_val = np.load('D:\HECHAOWEI\TFProject4\data2/val_labels_one_hot.npy')
    Y_test = np.load('D:\HECHAOWEI\TFProject4\data2/test_labels_one_hot.npy')

    Z_train = np.load('D:\HECHAOWEI\TFProject4\data2/train_snr.npy')
    Z_val = np.load('D:\HECHAOWEI\TFProject4\data2/val_snr.npy')
    Z_test = np.load('D:\HECHAOWEI\TFProject4\data2/test_snr.npy')

    # Define the number of modulations and SNRs
    num_modulations = Y_train.shape[1]  # Number of classes in one-hot encoding
    num_snrs_train = len(np.unique(Z_train))  # Unique SNR values in training set
    num_snrs_val = len(np.unique(Z_val))  # Unique SNR values in validation set

    # Define the number of samples per class and SNR
    samples_per_class_train = 6
    samples_per_class_val = 6

    def select_samples(X, Y, Z, num_modulations, samples_per_class, unique_snrs):
        X_selected = []
        Y_selected = []
        Z_selected = []

        for mod in range(num_modulations):
            for snr in unique_snrs:
                indices = np.where((np.argmax(Y, axis=1) == mod) & (Z == snr))[0]

                if len(indices) >= samples_per_class:
                    selected_indices = np.random.choice(indices, samples_per_class, replace=False)
                    X_selected.append(X[selected_indices])
                    Y_selected.append(Y[selected_indices])
                    Z_selected.append(Z[selected_indices])
                else:
                    print(f"Not enough samples for modulation {mod} and SNR {snr}. Available: {len(indices)}")

        if X_selected:
            X_selected = np.vstack(X_selected)
            Y_selected = np.vstack(Y_selected)
            Z_selected = np.hstack(Z_selected)

        return X_selected, Y_selected, Z_selected

    # Select samples for the training set
    unique_snrs_train = np.unique(Z_train)
    X_train_selected, Y_train_selected, Z_train_selected = select_samples(
        X_train, Y_train, Z_train, num_modulations, samples_per_class_train, unique_snrs_train)

    # Select samples for the validation set
    unique_snrs_val = np.unique(Z_val)
    X_val_selected, Y_val_selected, Z_val_selected = select_samples(
        X_val, Y_val, Z_val, num_modulations, samples_per_class_val, unique_snrs_val)

    # Assign the selected samples back to the original variables
    X_train = X_train_selected
    Y_train = Y_train_selected
    Z_train = Z_train_selected

    X_val = X_val_selected
    Y_val = Y_val_selected
    Z_val = Z_val_selected

    X_train,X_val,X_test = to_amp_phase(X_train,X_val,X_test,512)

    X_train = X_train[:,:maxlen,:]
    X_val = X_val[:,:maxlen,:]
    X_test = X_test[:,:maxlen,:]
    Z_train = Z_train.reshape(-1, 1)
    Z_val = Z_val.reshape(-1, 1)
    Z_test = Z_test.reshape(-1, 1)
    X_train = norm_pad_zeros(X_train,maxlen)
    X_val = norm_pad_zeros(X_val,maxlen)
    X_test = norm_pad_zeros(X_test,maxlen)

    return (X_train,Y_train,Z_train),(X_val,Y_val,Z_val),(X_test,Y_test,Z_test)

if __name__ == '__main__':
    (X_train,Y_train,Z_train),(X_val,Y_val,Z_val),(X_test,Y_test,Z_test) = load_data()
