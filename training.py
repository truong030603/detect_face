from deepface import DeepFace
import os
import pickle
import numpy as np
import cv2

def extract_feature(image_path):
    """
    Trích xuất vector đặc trưng từ ảnh sử dụng DeepFace.
    """
    embedding = DeepFace.represent(img_path=image_path, model_name="VGG-Face", enforce_detection=False)
    return embedding[0]["embedding"]

def prepare_data(data_dir):
    """
    Duyệt qua thư mục dữ liệu, trích xuất vector đặc trưng và gắn nhãn.
    """
    features = []
    labels = []

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue

        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            try:
                feature = extract_feature(image_path)
                features.append(feature)
                labels.append(label)
            except:
                print(f"Không thể xử lý ảnh: {image_path}")

    return np.array(features), np.array(labels)

# Đường dẫn thư mục chứa dữ liệu
data_dir = "rawFaceData"

# Trích xuất đặc trưng và lưu vào file
print("Trích xuất đặc trưng từ tập dữ liệu...")
features, labels = prepare_data(data_dir)
print(f"Đặc trưng shape: {features.shape}, Số lượng nhãn: {len(labels)}")

# Lưu đặc trưng và nhãn
with open("deepface_features.pkl", "wb") as f:
    pickle.dump((features, labels), f)
print("Đã lưu đặc trưng và nhãn vào file 'deepface_features.pkl'")
