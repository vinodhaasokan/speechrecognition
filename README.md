# speechrecognition
🕒 Speech Recognition Timer Control
This project is a simple speech-controlled timer app that uses machine learning to understand voice commands and control the timer accordingly. It includes a Speech-to-Text module and a complete speech recognition pipeline trained using the Google Speech Commands Dataset.

🎯 Project Objective
Use machine learning to classify short audio commands like "go" and "stop".

Start and stop a timer based on spoken input.

Include a speech-to-text component for recognizing single-word audio inputs.

📁 Modules Included
🎤 Voice-Controlled Timer

Say "go" → timer starts

Say "stop" → timer stops and shows elapsed time

📝 Speech-to-Text (Single-Word)

Upload or record a 1-second audio clip

Model predicts the spoken word

Displays predicted text (e.g., "go", "stop", "left", etc.)

🗃️ Dataset Used
Dataset: Google Speech Commands Dataset

Audio Format: 1-second .wav files

Used Labels:

go, stop → Timer Control

Others (optional): left, right, yes, no → For Speech-to-Text

🛠️ Technologies Used
Language: Python

Libraries:

Librosa – Audio feature extraction (MFCC)

NumPy, Pandas – Data manipulation

TensorFlow / Keras – Deep learning model

Sounddevice, Tkinter (or CLI) – UI and audio input

Matplotlib – Accuracy and loss plots

🧠 Speech Recognition Pipeline
Data Collection

Use Google Speech Commands Dataset

Extract go and stop for timer

(Optional) Add more classes for speech-to-text

Preprocessing

Convert .wav audio to MFCC features using Librosa

Normalize and reshape for CNN input

Model Architecture

Convolutional Neural Network (CNN)

Input: MFCC spectrogram

Output: Word class (e.g., “go”, “stop”)

Training

Use Keras with accuracy/loss tracking

Save the best model for inference

Inference (Real-time / File)

Record audio or upload a file

Extract MFCC and predict using the model

Return the recognized word or control the timer

🚀 How to Run
Clone the Repository

bash
Copy
Edit
git clone https://github.com/vinodhaasokan/speech-timer.git
cd speech-timer
Install Requirements

bash
Copy
Edit
pip install -r requirements.txt
Train the Model (or use pre-trained)

bash
Copy
Edit
python train_model.py
Run the Timer App

bash
Copy
Edit
python timer_control.py
Test Speech-to-Text

bash
Copy
Edit
python predict_word.py --file sample.wav
📊 Project Folder Structure
graphql
Copy
Edit
speech-timer/
├── dataset/                  # Google Speech Commands subset
├── model/                    # Trained CNN model (.h5)
├── train_model.py            # Model training script
├── timer_control.py          # Voice-controlled timer logic
├── predict_word.py           # Speech-to-text prediction from file
├── utils.py                  # Feature extraction and helper functions
├── requirements.txt
└── README.md
✅ Features
Start/stop timer with real voice

Single-word speech-to-text recognition

Works offline (after training)

Simple and modular code

⚠️ Limitations
Works best in quiet environments

Only supports short 1-second commands

Limited vocabulary unless more classes are added

🔮 Future Enhancements
Add continuous speech recognition (not just single-word)

Improve background noise filtering

Add pause/reset and voice feedback features

👩‍💻 Developers
Vinodha – Timer module, training, integration
Documentation, speech-to-text pipeline

📚 References
Google Speech Commands Dataset

TensorFlow Audio Tutorial

Librosa Docs
