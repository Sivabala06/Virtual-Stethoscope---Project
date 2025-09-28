## Virtual Stethoscope – AI for Heart Murmur Detection

This project is about building an AI-powered "virtual stethoscope" that listens to heart sounds (PCG) and predicts whether they are normal or have a murmur.
The goal is to support early cardiac screening and help clinicians make faster decisions, especially in areas where specialists are not easily available.

## Objective

Develop an AI model that classifies heart sounds as Normal or Murmur

Reduce missed diagnoses and assist doctors in screening large numbers of patients quickly

## Dataset

Source: PhysioNet Heart Sound Dataset (Challenge 2016)

Classes: Normal and Murmur

## Preprocessing:

Resampled to 2 kHz

Segmented into 3–5 second clips

Converted into Mel-spectrograms for CNN input

## Model

Input: Mel-spectrogram images

Backbone: CNN (EfficientNetB0)

Loss function: Binary Cross-Entropy

Optimizer: Adam

Metrics: Accuracy, Sensitivity, Specificity

## Results

Accuracy: 83%

Sensitivity: Good at detecting murmurs

Specificity: Sometimes predicts murmur when sound is actually normal

The model shows some bias — it tends to classify normal sounds as murmur more often than it should (false positives).
This is likely due to dataset imbalance and could be improved with data augmentation or fine-tuning.

## Known Limitations

False positives (normal predicted as murmur)

Dataset slightly imbalanced (fewer murmur samples than normal samples)

Model may not generalize perfectly to different recording conditions

## Future Work

Improve dataset balance with augmentation or SMOTE

Adjust decision threshold to reduce false positives

Extend to multi-class classification (different types of murmurs)

Add explainability with Grad-CAM to show which parts of the sound the model uses

## Requirements

Python 3.9 or newer

TensorFlow / Keras

NumPy, Pandas

Librosa

Matplotlib
