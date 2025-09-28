import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Check your dataset structure
root_dir = '/kaggle/input/heart-sound-database/heart_sound'

print("Dataset structure:")
for split in ['train', 'val']:
    for label in ['healthy', 'unhealthy']:
        path = os.path.join(root_dir, split, label)
        num_files = len([f for f in os.listdir(path) if f.endswith('.wav')])
        print(f"{split}/{label}: {num_files} files")

# Load a sample file to check audio properties
sample_file = os.path.join(root_dir, 'train', 'healthy', os.listdir(os.path.join(root_dir, 'train', 'healthy'))[0])
audio, sr = librosa.load(sample_file, sr=None)

print(f"\nSample audio properties:")
print(f"Duration: {len(audio)/sr:.2f} seconds")
print(f"Sample rate: {sr} Hz")
print(f"Audio shape: {audio.shape}")


# %% [code] {"execution":{"iopub.status.busy":"2025-09-15T05:11:04.715479Z","iopub.execute_input":"2025-09-15T05:11:04.715826Z","iopub.status.idle":"2025-09-15T05:11:04.742233Z","shell.execute_reply.started":"2025-09-15T05:11:04.715804Z","shell.execute_reply":"2025-09-15T05:11:04.741009Z"}}
import scipy.signal
import librosa
import numpy as np

def preprocess_heart_audio(filepath, target_sr=1000):
    """
    Load and preprocess heart sound audio
    """
    # Load audio
    audio, orig_sr = librosa.load(filepath, sr=None)
    
    # Bandpass filter for heart sounds (25-400 Hz)
    def bandpass_filter(signal, sr, lowcut=25, highcut=400, order=4):
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = min(highcut / nyq, 0.99)  # Ensure < 1.0
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        return scipy.signal.filtfilt(b, a, signal)
    
    # Apply bandpass filter
    audio_filtered = bandpass_filter(audio, orig_sr)
    
    # Resample to target sample rate
    if orig_sr != target_sr:
        audio_filtered = librosa.resample(audio_filtered, orig_sr=orig_sr, target_sr=target_sr)
    
    # Normalize audio
    audio_normalized = audio_filtered / (np.max(np.abs(audio_filtered)) + 1e-8)
    
    return audio_normalized, target_sr

# Test the preprocessing function
sample_file = os.path.join(root_dir, 'train', 'healthy', os.listdir(os.path.join(root_dir, 'train', 'healthy'))[0])
processed_audio, new_sr = preprocess_heart_audio(sample_file)

print(f"Original duration: {35.67:.2f} seconds")
print(f"Processed duration: {len(processed_audio)/new_sr:.2f} seconds")
print(f"New sample rate: {new_sr} Hz")
print(f"Audio range: [{processed_audio.min():.3f}, {processed_audio.max():.3f}]")


# %% [code] {"execution":{"iopub.status.busy":"2025-09-15T05:11:54.487104Z","iopub.execute_input":"2025-09-15T05:11:54.487431Z","iopub.status.idle":"2025-09-15T05:11:54.502716Z","shell.execute_reply.started":"2025-09-15T05:11:54.487409Z","shell.execute_reply":"2025-09-15T05:11:54.501472Z"}}
def segment_audio(audio, sr, segment_length=5.0, overlap=0.5):
    """
    Segment long audio into fixed-length chunks with overlap
    
    Args:
        audio: preprocessed audio array
        sr: sample rate
        segment_length: length of each segment in seconds
        overlap: overlap fraction (0.0-1.0)
    
    Returns:
        List of audio segments
    """
    segment_samples = int(segment_length * sr)
    hop_samples = int(segment_samples * (1 - overlap))
    
    segments = []
    start = 0
    
    while start + segment_samples <= len(audio):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
        start += hop_samples
    
    return segments

# Test segmentation
processed_audio, sr = preprocess_heart_audio(sample_file)
segments = segment_audio(processed_audio, sr, segment_length=5.0, overlap=0.5)

print(f"Original audio length: {len(processed_audio)/sr:.2f} seconds")
print(f"Number of 5-second segments: {len(segments)}")
print(f"Each segment length: {len(segments[0])/sr:.2f} seconds")
print(f"Segment shape: {segments[0].shape}")


# %% [code] {"execution":{"iopub.status.busy":"2025-09-15T05:12:41.971609Z","iopub.execute_input":"2025-09-15T05:12:41.971980Z","iopub.status.idle":"2025-09-15T05:12:42.406786Z","shell.execute_reply.started":"2025-09-15T05:12:41.971952Z","shell.execute_reply":"2025-09-15T05:12:42.405799Z"}}
import librosa
import numpy as np

def extract_mel_spectrogram(audio_segment, sr=1000, n_mels=128, n_fft=512, hop_length=256):
    """
    Extract log-mel spectrogram from audio segment
    
    Args:
        audio_segment: 1D audio array
        sr: sample rate
        n_mels: number of mel bands
        n_fft: FFT window size
        hop_length: hop length for STFT
    
    Returns:
        Log-mel spectrogram (time x frequency)
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_segment,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=sr//2  # Nyquist frequency
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Transpose to get (time, frequency) shape
    return log_mel_spec.T

# Test feature extraction
test_segment = segments[0]
spectrogram = extract_mel_spectrogram(test_segment)

print(f"Audio segment shape: {test_segment.shape}")
print(f"Spectrogram shape: {spectrogram.shape}")
print(f"Spectrogram range: [{spectrogram.min():.2f}, {spectrogram.max():.2f}] dB")

# Visualize the spectrogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.imshow(spectrogram.T, aspect='auto', origin='lower', cmap='viridis')
plt.ylabel('Mel Frequency Bins')
plt.xlabel('Time Frames')
plt.title('Log-Mel Spectrogram')
plt.colorbar(label='Power (dB)')
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2025-09-15T05:13:59.212595Z","iopub.execute_input":"2025-09-15T05:13:59.212921Z","iopub.status.idle":"2025-09-15T05:15:42.096285Z","shell.execute_reply.started":"2025-09-15T05:13:59.212899Z","shell.execute_reply":"2025-09-15T05:15:42.094902Z"}}
def process_dataset(root_dir, target_sr=1000, segment_length=5.0, overlap=0.5):
    """
    Process entire dataset: load, preprocess, segment, and extract features
    """
    all_spectrograms = []
    all_labels = []
    
    # Process both train and val sets
    for split in ['train', 'val']:
        for label_idx, label_name in enumerate(['healthy', 'unhealthy']):
            label_dir = os.path.join(root_dir, split, label_name)
            
            print(f"Processing {split}/{label_name}...")
            file_count = 0
            
            for filename in os.listdir(label_dir):
                if filename.endswith('.wav'):
                    filepath = os.path.join(label_dir, filename)
                    
                    # Preprocess audio
                    audio, sr = preprocess_heart_audio(filepath, target_sr)
                    
                    # Segment audio
                    segments = segment_audio(audio, sr, segment_length, overlap)
                    
                    # Extract spectrograms for each segment
                    for segment in segments:
                        spectrogram = extract_mel_spectrogram(segment, sr)
                        all_spectrograms.append(spectrogram)
                        all_labels.append(label_idx)
                    
                    file_count += 1
                    if file_count % 100 == 0:
                        print(f"  Processed {file_count} files...")
    
    return np.array(all_spectrograms), np.array(all_labels)

# Process the dataset (this will take a few minutes)
print("Starting complete dataset processing...")
X_all, y_all = process_dataset(root_dir)

print(f"\nDataset processing complete!")
print(f"Total spectrograms: {X_all.shape[0]}")
print(f"Spectrogram shape: {X_all.shape[1:]}")
print(f"Label distribution: {np.bincount(y_all)}")


# %% [code] {"execution":{"iopub.status.busy":"2025-09-15T05:18:44.474638Z","iopub.execute_input":"2025-09-15T05:18:44.474992Z","iopub.status.idle":"2025-09-15T05:18:44.493562Z","shell.execute_reply.started":"2025-09-15T05:18:44.474969Z","shell.execute_reply":"2025-09-15T05:18:44.492209Z"}}
# Number of train segments
n_train_segments = 19608

# Split features and labels
X_train = X_all[:n_train_segments]
y_train = y_all[:n_train_segments]
X_val   = X_all[n_train_segments:]
y_val   = y_all[n_train_segments:]

print("Train set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)


# %% [code] {"execution":{"iopub.status.busy":"2025-09-15T05:19:41.631493Z","iopub.execute_input":"2025-09-15T05:19:41.632276Z","iopub.status.idle":"2025-09-15T05:19:41.833250Z","shell.execute_reply.started":"2025-09-15T05:19:41.632240Z","shell.execute_reply":"2025-09-15T05:19:41.832389Z"}}
import tensorflow as tf
from tensorflow.keras import layers, models

# Add channel dimension
X_train = X_train[..., np.newaxis]  # shape: (19608, 20, 128, 1)
X_val   = X_val[...,   np.newaxis]  # shape: (6963, 20, 128, 1)

input_shape = X_train.shape[1:]  # (20, 128, 1)
num_classes = 2

inputs = layers.Input(shape=input_shape)

# CNN feature extractor
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((1,2))(x)     # pool only frequency
x = layers.Dropout(0.3)(x)

x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((1,2))(x)
x = layers.Dropout(0.3)(x)

# Collapse frequency dimension, preserve time frames
# New shape: (batch, time_steps, features)
time_steps = x.shape[1]
features  = x.shape[2] * x.shape[3]
x = layers.Reshape((time_steps, features))(x)

# LSTM layers
x = layers.LSTM(128, return_sequences=True)(x)
x = layers.Dropout(0.3)(x)
x = layers.LSTM(64)(x)
x = layers.Dropout(0.3)(x)

# Dense output
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# %% [code] {"execution":{"iopub.status.busy":"2025-09-15T05:37:08.764068Z","iopub.execute_input":"2025-09-15T05:37:08.764434Z","iopub.status.idle":"2025-09-15T06:14:52.732287Z","shell.execute_reply.started":"2025-09-15T05:37:08.764410Z","shell.execute_reply":"2025-09-15T06:14:52.731395Z"}}
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weights_dict
)


# %% [code] {"execution":{"iopub.status.busy":"2025-09-15T06:18:19.542972Z","iopub.execute_input":"2025-09-15T06:18:19.543362Z","iopub.status.idle":"2025-09-15T06:18:45.270926Z","shell.execute_reply.started":"2025-09-15T06:18:19.543337Z","shell.execute_reply":"2025-09-15T06:18:45.269629Z"}}
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f}")

y_val_pred = model.predict(X_val).argmax(axis=1)
from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Healthy','Unhealthy']))

cm = confusion_matrix(y_val, y_val_pred)


# %% [code] {"execution":{"iopub.status.busy":"2025-09-15T06:18:51.949235Z","iopub.execute_input":"2025-09-15T06:18:51.950156Z","iopub.status.idle":"2025-09-15T06:18:52.133403Z","shell.execute_reply.started":"2025-09-15T06:18:51.950131Z","shell.execute_reply":"2025-09-15T06:18:52.132384Z"}}
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy','Unhealthy'],
            yticklabels=['Healthy','Unhealthy'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
