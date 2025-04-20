import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


model = load_model('D:/coding/project/models/natural_images_model.h5')

class_labels = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

def predict_image(image_path):

    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)
    class_label = class_labels[class_idx[0]] 

    return class_label

image_path = "D:/coding/project/test/test.jpeg"
predicted_class = predict_image(image_path)
print(f"Predicted Class: {predicted_class}")
