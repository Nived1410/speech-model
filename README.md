# speech-model
# End-to-End Speech Emotion Recognition Using BLSTMs with Self-Attention and Multi-Domain Training

![Project Architecture](https://github.com/raulsteleac/Speech_Emotion_Recognition/blob/master/Sistem_Diagram.jpeg?raw=true)

This repository contains the source code for a Speech Emotion Recognition (SER) system built with TensorFlow 1.15. The project includes modules for feature extraction, classification, GUI interaction, real-time audio recording, and statistical visualization.

Developed as an undergraduate thesis project, this system serves as a foundation for further improvements in speech emotion analysis.

---

## Key Features

- **End-to-end deep learning model** for emotion recognition from speech.
- **Feature extraction** with two convolutional layers (CNN) operating on Mel-spectrogram inputs.
- **Classification** using two bidirectional LSTM (BLSTM) cells enhanced by a self-attention layer to focus on emotionally relevant frames.
- **Graphical User Interface (GUI)** built with PyQt5 for training, inference, and visualization of accuracy and confusion matrices.
- **Supports live audio recording and pre-recorded audio classification.**
- Multi-domain training strategy to improve generalization across datasets.

---

## Project Structure

| Folder / File            | Description                                                                                 |
|-------------------------|---------------------------------------------------------------------------------------------|
| `SER.py`                | Main entry point launching the GUI interface for training and inference                     |
| `model.py`              | Implementation of the core deep learning classification model                               |
| `feature_extractors/`   | Modules for both handcrafted and end-to-end feature extraction (only end-to-end used)       |
| `graphics/`             | GUI source code built using PyQt5                                                           |
| `recording/`            | Audio recording functionality utilizing PyAudio                                            |
| `best_current_model/`   | Contains the highest performing pre-trained model files                                    |
| `README.md`             | Project overview and documentation                                                         |
| `requirements.txt`      | Python dependencies and versions required for setup                                        |

---

## Technology Stack

- **TensorFlow 1.15** - Model building and training
- **Librosa** - Audio processing and Mel-spectrogram extraction
- **PyQt5** - Graphical User Interface design
- **PyAudio** - Audio I/O and microphone recording
- **qdarkgraystyle** - GUI dark theme styling
- **webrtcvad** - Voice activity detection for real-time audio filtering

---

## Getting Started

### Environment Setup

1. Create and activate a Python environment (recommended):

    ```
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    ```

2. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

   *Make sure TensorFlow 1.15 is installed (`pip install tensorflow==1.15`) as newer versions are not compatible.*

### Running the Application

Launch the GUI to start training or inference:

