# Import libraries
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Load the VGG16 model pretrained on ImageNet
model = VGG16(weights='imagenet')

# Load an image and resize to 224x224 (VGG16 input size)
# img_path = "dog.jpg"   # replace with your image path

# --- FIX: Using a placeholder image URL for demonstration ---
# Original problematic URL: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Labrador_Retriever_dog.jpg/1200px-Labrador_Retriever_dog.jpg"
img_url = "https://images.pexels.com/photos/2071882/pexels-photo-2071882.jpeg" # Using a more robust sample image URL
response = requests.get(img_url)
response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
img = Image.open(BytesIO(response.content))
img = img.resize((224, 224))

# Convert image to array
x = image.img_to_array(img)

# Add batch dimension (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# Preprocess for VGG16
x = preprocess_input(x)

# Make prediction
preds = model.predict(x)
plt.imshow(img)
plt.axis('off')
plt.savefig("output.png")


# Decode predictions (top 3)
print('Predicted:', decode_predictions(preds, top=3)[0])                                   
