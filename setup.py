import os
import cv2
import dlib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# 1. Trích xuất đặc trưng khuôn mặt
def extract_face_features(image_path, face_detector, shape_predictor, face_recognizer):
    """
    Trích xuất đặc trưng 128-d của khuôn mặt từ ảnh.
    Args:
        image_path (str): Đường dẫn tới ảnh.
        face_detector: Bộ phát hiện khuôn mặt của dlib.
        shape_predictor: Bộ dự đoán hình dáng khuôn mặt.
        face_recognizer: Bộ nhận diện khuôn mặt của dlib.
    Returns:
        numpy.ndarray: Đặc trưng khuôn mặt (128-d).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces) == 0:
        raise ValueError(f"Không tìm thấy khuôn mặt trong ảnh: {image_path}")

    # Lấy khuôn mặt đầu tiên
    face = faces[0]
    shape = shape_predictor(gray, face)
    face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

# 2. Chuẩn bị dữ liệu
def prepare_training_data(data_dir, face_detector, shape_predictor, face_recognizer):
    """
    Trích xuất đặc trưng khuôn mặt từ thư mục chứa ảnh và gắn nhãn.
    Args:
        data_dir (str): Thư mục chứa ảnh được tổ chức theo người (rawfacedata).
        face_detector, shape_predictor, face_recognizer: Các mô-đun của dlib.
    Returns:
        list, list: Đặc trưng khuôn mặt và nhãn.
    """
    features = []
    labels = []

    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            try:
                face_descriptor = extract_face_features(image_path, face_detector, shape_predictor, face_recognizer)
                features.append(face_descriptor)
                labels.append(person_name)
            except ValueError as e:
                print(e)

    return features, labels

# 3. Huấn luyện SVM
def train_svm(features, labels):
    """
    Huấn luyện mô hình SVM từ đặc trưng và nhãn.
    Args:
        features (list): Danh sách vector đặc trưng khuôn mặt.
        labels (list): Danh sách nhãn tương ứng.
    Returns:
        SVC, LabelEncoder: Mô hình SVM đã huấn luyện và bộ mã hóa nhãn.
    """
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)

    svm = SVC(kernel='linear', probability=True)
    svm.fit(features, labels_encoded)

    return svm, encoder

# 4. Test với camera
def run_camera_recognition(svm, encoder, face_detector, shape_predictor, face_recognizer):
    """
    Chạy nhận diện khuôn mặt trực tiếp từ camera.
    Args:
        svm: Mô hình SVM đã huấn luyện.
        encoder: Bộ mã hóa nhãn.
        face_detector, shape_predictor, face_recognizer: Các mô-đun của dlib.
    """
    cap = cv2.VideoCapture(0)  # Mở camera (ID = 0 là camera mặc định)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        for face in faces:
            shape = shape_predictor(gray, face)
            face_descriptor = np.array(face_recognizer.compute_face_descriptor(frame, shape))

            # Dự đoán danh tính
            prediction = svm.predict([face_descriptor])
            name = encoder.inverse_transform(prediction)[0]

            # Vẽ bounding box và hiển thị tên
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Hiển thị frame
        cv2.imshow("Face Recognition", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main script
if __name__ == "__main__":
    # Dlib models
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    # 1. Chuẩn bị dữ liệu
    data_dir = "rawfacedata"  # Thư mục chứa ảnh
    print("Trích xuất đặc trưng từ dữ liệu huấn luyện...")
    features, labels = prepare_training_data(data_dir, face_detector, shape_predictor, face_recognizer)

    # 2. Huấn luyện mô hình
    print("Huấn luyện mô hình SVM...")
    svm, encoder = train_svm(features, labels)
    print("Mô hình đã được huấn luyện thành công!")

    # 3. Test bằng camera
    print("Mở camera để nhận diện...")
    run_camera_recognition(svm, encoder, face_detector, shape_predictor, face_recognizer)
