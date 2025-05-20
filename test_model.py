from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load your saved model
model = load_model("accident_classifier_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    pred_prob = model.predict(img_array)[0][0]  # sigmoid output

    pred_class = 1 if pred_prob > 0.5 else 0

    # Adjust this based on your training data.class_indices output
    class_names = {0: 'non_accident', 1: 'accident'}

    print(f"Image: {os.path.basename(img_path)}")
    print(f"Raw model output (probability): {pred_prob:.4f}")
    print(f"Predicted class index: {pred_class}")
    print(f"Predicted class label: {class_names[pred_class]}")

if __name__ == "__main__":
    # Test on a sample image (change to your own test image)
    test_image_path = "D:/videodata/data/test/NonAccident/test28_4.jpg"
    predict_image(test_image_path)
