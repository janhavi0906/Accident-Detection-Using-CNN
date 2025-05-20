Accident Detection Classifier
A Convolutional Neural Network (CNN) model for classifying road images into Accident or Non-Accident categories.
Includes support for single image prediction and real-time webcam detection using OpenCV.


videodata/
│
├── data/
│   ├── train/
│   │   ├── Accident/
│   │   └── Non_Accident/
│   └── val/
│       ├── Accident/
│       └── Non_Accident/
│
├── train_model.py          # Train and save CNN model
├── predict_image.py        # Predict class of a single image
├── accident_classifier_model.h5  # Trained model file
└── README.md
