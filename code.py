import os
import regex as re
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import keras_tuner as kt
from keras.models import Sequential
import numpy as np
from scipy.signal import butter, filtfilt
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

combined_save_loc = r"combine_data.npz"
chunk = np.load(combined_save_loc)
shapes = np.random.permutation(chunk['series1'].shape[0])
train_wav = chunk['series1'][shapes,:,:]
train_label = chunk['series2'][shapes,:]

# Assume train_wav (42048, 200, 1) and train_label (42048, 1) are already loaded as numpy arrays.
# For example:
# train_wav = np.load('train_wav.npy')   # shape (42048, 200, 1)
# train_label = np.load('train_label.npy')   # shape (42048, 1)

# 1. Bandpass filter design for 0.1–10 Hz
fs = 20.0  # Sampling rate (Hz) – adjust if known, 20 Hz is assumed here for a 10s signal length&#8203;:contentReference[oaicite:3]{index=3}.
lowcut = 0.1
highcut = 10.0
order = 4

nyq = 0.5 * fs  # Nyquist frequency
low = lowcut / nyq
high = highcut / nyq if highcut < nyq else 0.9999  # avoid exactly 1.0 (Nyquist) to prevent instability

# Design a Butterworth bandpass (or high-pass if highcut is Nyquist)
if highcut >= 0.5 * fs:
    # Highcut at Nyquist – design as high-pass filter only
    b, a = butter(order, low, btype='high')
else:
    b, a = butter(order, [low, high], btype='band')

# Apply the bandpass filter to each signal (zero-phase filtering with filtfilt)
# Flatten the last dimension for filtering and filter along axis=1 (time axis)
signals = train_wav.squeeze(axis=-1)  # shape becomes (42048, 200)
filtered_signals = filtfilt(b, a, signals, axis=1)  # apply filter along each time-series row

# 2. Spectrogram parameters
n_fft = 64        # FFT window size (number of samples per FFT)
hop_length = 16   # hop length (stride) between successive FFT windows
n_mels = 32       # number of Mel frequency bands
fmin = lowcut     # 0.1 Hz lower bound
fmax = highcut    # 10 Hz upper bound

# Function to compute log-Mel spectrogram for one signal
def compute_log_mel_spectrogram(signal, sr):
    """
    Compute the log-mel spectrogram of a 1D `signal` using given sample rate `sr`.
    Returns a 2D array (n_mels x time_frames) in decibel units.
    """
    # Compute Mel-scaled power spectrogram
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, 
                                       n_mels=n_mels, fmin=fmin, fmax=fmax)
    # Convert power spectrogram (amplitude^2) to decibels (log scale)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

# 3. Compute spectrogram for all signals
spectrogram_list = []
for sig in filtered_signals:
    # Each `sig` is a 1D numpy array of length 200 (filtered time-series)
    S_db = compute_log_mel_spectrogram(sig, sr=fs)
    spectrogram_list.append(S_db)

spectrograms = np.array(spectrogram_list)  # shape: (42048, n_mels, time_frames)
# Add a channel dimension for CNN (as grayscale image)
spectrograms = spectrograms[..., np.newaxis]  # shape: (42048, n_mels, time_frames, 1)

print("Spectrogram dataset shape:", spectrograms.shape)
print("Label array shape:", train_label.shape)

# output -  Spectrogram dataset shape: (46721, 32, 13, 1)
# Label array shape: (46721, 1)



# Assume you already have:
# - spectrograms: numpy array of shape (42048, 32, 13, 1)
# - train_label: numpy array of shape (42048, 1)

# 1. Split the spectrogram data into training and testing sets.
#    Here we use an 80/20 split while stratifying to preserve the label distribution.
X_train, X_test, y_train, y_test = train_test_split(
    spectrograms, train_label, test_size=0.2, stratify=train_label, random_state=42
)

print("Training set shape:", X_train.shape, "Labels shape:", y_train.shape)
print("Test set shape:", X_test.shape, "Labels shape:", y_test.shape)

# 2. Build a VGG-style CNN model.
#    The input shape to the model is (32, 13, 1) as per your spectrogram dimensions.
input_shape = X_train.shape[1:]  # (32, 13, 1)

model = Sequential([
    # Block 1: Two convolutional layers followed by max pooling.
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),  # Reduces size: from (32, 13) to roughly (16, 6)

    # Block 2: Increase filters to capture more complex features.
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),  # Reduces size further, e.g., ~ (8, 3)

    # Block 3: Further deepen the network.
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),  # Final spatial dimension might be very small (~ (4, 1))

    # Flatten the feature maps and use dense layers for classification.
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Use dropout to reduce overfitting
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model with binary crossentropy loss (appropriate for 0/1 labels)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Show the model architecture summary.
model.summary()

# 3. Train the model.
epochs = 80
batch_size = 30

history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test)
)

# 4. Evaluate the model on the test set.
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


# Save the trained model in HDF5 format
model.save("vgg_spectrogram_classifier.h5")
print("Model saved as 'vgg_spectrogram_classifier.h5'")
