import os
import cv2
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pickle

def load_face_dataset(data_path):
    """Load và xử lý dataset từ thư mục"""
    face_data = []
    labels = []
    
    # Duyệt qua các thư mục con
    for person_name in os.listdir(data_path):
        person_dir = os.path.join(data_path, person_name)
        if os.path.isdir(person_dir):
            # Duyệt qua các ảnh trong thư mục của mỗi người
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                try:
                    # Load và resize ảnh về 224x224
                    img = load_img(img_path, target_size=(224, 224))
                    img_array = img_to_array(img)
                    
                    # Thêm vào danh sách
                    face_data.append(img_array)
                    labels.append(person_name)
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
    
    return np.array(face_data), np.array(labels)

def extract_features(model, face_data):
    """Trích xuất đặc trưng sử dụng VGG16"""
    # Tiền xử lý ảnh
    preprocessed_data = preprocess_input(face_data)
    
    # Trích xuất đặc trưng
    features = model.predict(preprocessed_data)
    # Làm phẳng đặc trưng để sử dụng với KNN
    features = features.reshape(features.shape[0], -1)
    return features

def train_face_recognition(raw_data_path):
    """Train mô hình nhận dạng khuôn mặt"""
    # Load VGG16 model, bỏ lớp fully connected cuối
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Load dataset
    print("Loading dataset...")
    face_data, labels = load_face_dataset(raw_data_path)
    
    # Trích xuất đặc trưng
    print("Extracting features...")
    features = extract_features(base_model, face_data)
    
    # Encode nhãn
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Train KNN classifier
    print("Training KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, encoded_labels)
    
    # Lưu mô hình và label encoder
    with open('face_recognition_model.pkl', 'wb') as f:
        pickle.dump((knn, label_encoder), f)
    
    return base_model, knn, label_encoder

def process_camera_frame(frame, vgg_model, knn, label_encoder):
    """Xử lý frame từ camera và nhận dạng khuôn mặt"""
    # Resize frame về 224x224
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    # Trích xuất đặc trưng
    features = extract_features(vgg_model, img)
    
    # Dự đoán
    pred = knn.predict(features)
    person_name = label_encoder.inverse_transform(pred)[0]
    
    return person_name

def run_face_recognition():
    # Train mô hình
    raw_data_path = "rawFaceData"  # Đường dẫn đến thư mục dataset
    vgg_model, knn, label_encoder = train_face_recognition(raw_data_path)
    
    # Khởi tạo camera
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Starting face recognition... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Phát hiện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            try:
                # Cắt vùng khuôn mặt và thêm padding
                face_frame = frame[max(0, y-30):min(frame.shape[0], y+h+30), 
                                 max(0, x-30):min(frame.shape[1], x+w+30)]
                
                # Nhận dạng khuôn mặt
                person_name = process_camera_frame(face_frame, vgg_model, knn, label_encoder)
                
                # Vẽ khung và tên
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, person_name, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            except Exception as e:
                print(f"Error processing face: {str(e)}")
        
        # Hiển thị frame
        cv2.imshow('Face Recognition', frame)
        
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_recognition()