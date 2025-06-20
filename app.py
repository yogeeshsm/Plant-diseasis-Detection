import os
import time
import cv2
import streamlit as st
import numpy as np
from numpy import argmax
from PIL import Image
from pathlib import Path
import tensorflow as tf
from keras.preprocessing import image
from utils import label_map_util
from utils import visualization_utils as vis_util

# --- CONSTANTS ---
DETECTION_MODEL_PATH = './object_detection/inference_graph/frozen_inference_graph.pb'
LABELMAP_PATH = './object_detection/training/labelmap.pbtxt'
CLASSIFICATION_MODEL_PATH = './object_classification/rps.h5'
UPLOAD_PATH = './uploads/temp_image.jpg'
NUM_CLASSES = 6
CLASS_LABELS = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy',
    'Cherry___healthy','Cherry___Powdery_mildew','Grape___Black_rot','Grape___Esca_Black_Measles','Grape___healthy',
    'Grape___Leaf_blight_Isariopsis_Leaf_Spot','Orange___Haunglongbing','Peach___Bacterial_spot','Peach___healthy',
    'Pepper_bell___Bacterial_spot','Pepper_bell___healthy','Potato___Early_blight','Potato___healthy','Potato___Late_blight',
    'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___healthy','Strawberry___Leaf_scorch'
]

# --- CACHED MODEL LOADERS ---
@st.cache_resource
def load_detection_model():
    detection_graph = tf.compat.v1.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(DETECTION_MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    return detection_graph, sess

@st.cache_resource
def load_classification_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(24, activation='softmax')
    ])
    model.load_weights(CLASSIFICATION_MODEL_PATH)
    return model

# --- UTILS ---
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def save_uploaded_image(uploaded_file):
    os.makedirs(os.path.dirname(UPLOAD_PATH), exist_ok=True)
    with open(UPLOAD_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return UPLOAD_PATH

# --- DETECTION ---
def run_detection(image_path, detection_graph, sess):
    label_map = label_map_util.load_labelmap(LABELMAP_PATH)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    image_np = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)
    return image_np

# --- CLASSIFICATION ---
def run_classification(image_path, model):
    img = image.load_img(image_path, target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    result = argmax(classes)
    return CLASS_LABELS[result]

# --- STREAMLIT APP ---
def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Plant Disease Detection & Classification")
    st.text("Built with Streamlit and Tensorflow")

    activities = ["About", "Plant Disease"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    enhance_type = st.sidebar.radio("Type", ["Detection", "Classification", "Treatment"])

    if choice == "About":
        intro_markdown = read_markdown_file("./doc/about.md")
        st.markdown(intro_markdown, unsafe_allow_html=True)

    # Load models once
    detection_graph, detection_sess = load_detection_model()
    classification_model = load_classification_model()

    if choice == "Plant Disease" and enhance_type == "Detection":
        st.header("Plant Disease Detection")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        st.markdown("* * *")
        if image_file is not None:
            img_path = save_uploaded_image(image_file)
            st.image(img_path, caption="Uploaded Image", use_column_width=True)
            if st.button('Process'):
                detected_img = run_detection(img_path, detection_graph, detection_sess)
                st.image(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)
                st.balloons()

    if choice == "Plant Disease" and enhance_type == "Classification":
        st.header("Plant Disease Classification")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        st.markdown("* * *")
        if image_file is not None:
            img_path = save_uploaded_image(image_file)
            st.image(img_path, caption="Uploaded Image", use_column_width=True)
            if st.button('Classify'):
                with st.spinner('Classifying...'):
                    result = run_classification(img_path, classification_model)
                    time.sleep(2)
                st.success(f"**Detected Disease:** {result}")
                st.balloons()

    if choice == "Plant Disease" and enhance_type == "Treatment":
        st.header("Plant Disease Treatment")
        data_markdown = read_markdown_file("./treatment/treatment.md")
        st.markdown(data_markdown, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
