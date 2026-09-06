import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
# PAGE CONFIGURATION
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide"
)
# CIFAR-10 CLASS NAMES
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
# PROJECT PATHS
MODEL_PATH = "best_exp2_cnn.keras"
EXAMPLES_DIR = "cifar10_examples"
# LOAD TRAINED MODEL
@st.cache_resource
def load_model():
    """
    Load the final Experiment 2 CNN model.
    """
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found: {MODEL_PATH}. "
            "Make sure best_exp2_cnn.keras is in the same folder as app.py."
        )
        st.stop()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


model = load_model()
# IMAGE PREPROCESSING
def preprocess_image(image):
    """
    Prepare uploaded image for the CIFAR-10 CNN.

    CIFAR-10 images:
    - RGB
    - 32 x 32 pixels
    - pixel values normalized to [0, 1]
    """

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to CIFAR-10 input size
    image = image.resize((32, 32))

    # Convert to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array
# PREDICTION FUNCTION
def predict_image(image):
    """
    Predict the CIFAR-10 class and return
    top predictions with confidence scores.
    """

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    probabilities = predictions[0]

    # Get indices of top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = []

    for index in top_indices:
        top_predictions.append(
            (
                CLASS_NAMES[index],
                float(probabilities[index])
            )
        )

    predicted_class = top_predictions[0][0]
    confidence = top_predictions[0][1]

    return predicted_class, confidence, top_predictions
# HEADER
st.title("🖼️ CIFAR-10 Image Classifier")

st.markdown(
    """
    Upload an image and the trained CNN model will predict
    which **CIFAR-10 class** it belongs to.
    """
)
# PROJECT INFORMATION
with st.expander(" About this Project", expanded=False):

    st.markdown(
        """
        ### CIFAR-10 Multi-Model Image Classification

        This project compares multiple machine learning and
        deep learning approaches for CIFAR-10 image classification.

        **Final selected model:**
        - Experiment 2 – Augmented CNN
        - Test Accuracy: **86.98%**
        - Input size: **32 × 32 RGB**
        - Number of classes: **10**

        The CNN was trained on the CIFAR-10 dataset and uses
        image augmentation and regularization techniques to
        improve generalization.
        """
    )

    st.info(
        "For the most reliable predictions, use images that "
        "visually resemble CIFAR-10 images: small, centered "
        "objects with a relatively simple background."
    )
# SIDEBAR
st.sidebar.header("📚 CIFAR-10 Classes")

for class_name in CLASS_NAMES:
    st.sidebar.write(f"• {class_name.capitalize()}")


st.sidebar.markdown("---")

st.sidebar.write(
    "**Model:** Experiment 2 – Augmented CNN"
)

st.sidebar.write(
    "**Test Accuracy:** 86.98%"
)

st.sidebar.write(
    "**Input:** 32 × 32 × 3"
)
# EXAMPLE IMAGES
st.header("Try Example CIFAR-10 Images")

st.write(
    "Use these provided examples to see how the trained model "
    "classifies CIFAR-10 images."
)
example_files = []

if os.path.exists(EXAMPLES_DIR):

    for class_name in CLASS_NAMES:

        image_path = os.path.join(
            EXAMPLES_DIR,
            f"{class_name}.png"
        )

        if os.path.exists(image_path):
            example_files.append(
                (class_name, image_path)
            )

if example_files:

    columns = st.columns(5)

    for i, (class_name, image_path) in enumerate(example_files):

        with columns[i % 5]:

            image = Image.open(image_path)

            st.image(
                image,
                caption=class_name.capitalize(),
                width="stretch"
            )

            if st.button(
                f"Predict {class_name.capitalize()}",
                key=f"example_{class_name}"
            ):

                predicted_class, confidence, top_predictions = (
                    predict_image(image)
                )

                st.success(
                    f"Prediction: {predicted_class.capitalize()}"
                )

                st.write(
                    f"Confidence: {confidence * 100:.2f}%"
                )
else:

    st.warning(
        "Example images were not found. "
        "Make sure the 'examples' folder contains the "
        "10 CIFAR-10 example PNG files."
    )
# IMAGE UPLOAD SECTION
st.markdown("---")

st.header("📤 Upload Your Own Image")

st.write(
    """
    Upload a JPG, JPEG or PNG image containing an object such as
    an airplane, automobile, bird, cat, deer, dog, frog, horse,
    ship or truck.
    """
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)
# PROCESS UPLOADED IMAGE
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    # ORIGINAL IMAGE
    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            width="stretch"
        )

        st.caption(
            f"Original size: {image.width} × {image.height} pixels"
        )
    # PREDICTION
    with col2:

        st.subheader("Prediction")

        with st.spinner("Analyzing image..."):

            predicted_class, confidence, top_predictions = (
                predict_image(image)
            )

        st.success(
            f"### {predicted_class.capitalize()}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )
        # CONFIDENCE MESSAGE
        if confidence >= 0.80:

            st.success(
                "The model has relatively high confidence "
                "in this prediction."
            )

        elif confidence >= 0.50:

            st.warning(
                "The model has moderate confidence. "
                "The image may contain visual characteristics "
                "shared by multiple CIFAR-10 classes."
            )

        else:

            st.warning(
                "The model has low confidence. "
                "Try a clearer image containing a single "
                "centered object."
            )
    # TOP 3 PREDICTIONS
    st.markdown("---")

    st.subheader("📊 Top 3 Predictions")

    for class_name, probability in top_predictions:

        st.write(
            f"**{class_name.capitalize()}** — "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            min(probability, 1.0)
        )
    # MODEL INPUT INFORMATION
    with st.expander("How the image is processed"):

        st.write(
            """
            The uploaded image goes through the following
            preprocessing pipeline:
            """
        )

        st.markdown(
            """
            1. Convert image to RGB
            2. Resize to **32 × 32 pixels**
            3. Convert pixels to numerical values
            4. Normalize pixel values to **0–1**
            5. Add the batch dimension
            6. Pass the image to the trained CNN
            7. Generate probabilities for all 10 CIFAR-10 classes
            8. Display the top prediction and top 3 predictions
            """
        )
# FOOTER
st.markdown("---")

st.caption(
    "CIFAR-10 Image Classification • "
    "Experiment 2 – Augmented CNN • "
    "86.98% Test Accuracy"
)