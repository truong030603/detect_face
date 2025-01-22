from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Đọc đặc trưng và nhãn từ file
with open("deepface_features.pkl", "rb") as f:
    features, labels = pickle.load(f)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Huấn luyện SVM
print("Huấn luyện mô hình SVM...")
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
print("Hoàn tất huấn luyện.")

# Đánh giá mô hình
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Lưu mô hình SVM
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)
print("Đã lưu mô hình SVM vào file 'svm_model.pkl'")
