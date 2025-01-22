import cv2
from deepface import DeepFace
import pickle

def classify_frame(frame, svm_model):
    """
    Phân loại khung hình từ camera bằng SVM.
    """
    # Lưu tạm thời khung hình
    temp_path = "temp_frame.jpg"
    cv2.imwrite(temp_path, frame)

    # Trích xuất đặc trưng
    try:
        embedding = DeepFace.represent(img_path=temp_path, model_name="VGG-Face", enforce_detection=False)
        feature = embedding[0]["embedding"]
    except:
        return "Không thể nhận diện"

    # Dự đoán bằng SVM
    prediction = svm_model.predict([feature])
    return prediction[0]

# Đọc mô hình SVM đã lưu
with open("svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

# Mở camera
cap = cv2.VideoCapture(0)

print("Đang mở camera. Nhấn 'q' để thoát.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc camera.")
        break

    # Phân loại khung hình
    label = classify_frame(frame, svm)

    # Hiển thị kết quả lên khung hình
    cv2.putText(frame, f"Detected: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Face Recognition", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
