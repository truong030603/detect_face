import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Khởi tạo Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def extract_features_mediapipe(image):
    """
    Trích xuất 468 điểm mốc từ khuôn mặt bằng Mediapipe.
    """
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    # Lấy khuôn mặt đầu tiên
    face_landmarks = results.multi_face_landmarks[0]
    # Trích xuất (x, y, z) cho mỗi điểm mốc
    keypoints = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
    return np.array(keypoints).flatten()  # Chuyển thành mảng 1D

def prepare_training_data(data_dir):
    """
    Trích xuất đặc trưng từ thư mục chứa ảnh.
    """
    features = []
    labels = []

    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Trích xuất đặc trưng
            feature = extract_features_mediapipe(image)
            if feature is not None:
                features.append(feature)
                labels.append(person_name)

    return features, labels

def compare_features(new_feature, saved_features, saved_labels, threshold=0.5):
    """
    So sánh đặc trưng từ camera với các đặc trưng đã lưu.
    """
    distances = np.linalg.norm(saved_features - new_feature, axis=1)
    min_distance = np.min(distances)
    if min_distance < threshold:
        return saved_labels[np.argmin(distances)]
    return "Unknown"

def run_camera_recognition(saved_features, saved_labels):
    """
    Nhận diện khuôn mặt từ camera và so sánh với đặc trưng đã lưu.
    """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Trích xuất đặc trưng từ camera
        feature = extract_features_mediapipe(frame)
        if feature is not None:
            name = compare_features(feature, saved_features, saved_labels)

            # Hiển thị kết quả
            cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị frame
        cv2.imshow("Face Recognition", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Thư mục dữ liệu
    data_dir = "rawFaceData"

    # Trích xuất và lưu đặc trưng
    features, labels = prepare_training_data(data_dir)
    features = np.array(features)
    with open("mediapipe_features.pkl", "wb") as f:
        pickle.dump((features, labels), f)

    # Nạp lại đặc trưng
    with open("mediapipe_features.pkl", "rb") as f:
        saved_features, saved_labels = pickle.load(f)

    # Nhận diện bằng camera
    run_camera_recognition(saved_features, saved_labels)
