import tensorflow as tf
import gradio as gr
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("rice_leaf_model.h5")

IMG_SIZE = (128, 128)
CLASS_NAMES = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]

def predict(img):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    return {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Rice Leaf Disease Detection",
    description="Upload a rice leaf image to predict the disease."
)

if __name__ == "__main__":
    demo.launch()
