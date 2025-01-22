import pickle

def check_pkl_file(pkl_file_path):
    """
    Đọc file .pkl và kiểm tra số lượng đặc trưng và nhãn dán.
    Args:
        pkl_file_path (str): Đường dẫn tới file .pkl.
    """
    try:
        # Đọc file .pkl
        with open(pkl_file_path, "rb") as f:
            features, labels = pickle.load(f)
        
        # Kiểm tra số lượng đặc trưng và nhãn
        num_features = len(features)
        num_labels = len(labels)

        print(f"Số lượng đặc trưng: {num_features}")
        print(f"Số lượng nhãn: {num_labels}")
        
        # Kiểm tra chi tiết nhãn
        unique_labels = set(labels)
        print(f"Số lượng nhãn duy nhất: {len(unique_labels)}")
        print(f"Các nhãn duy nhất: {unique_labels}")

        # Hiển thị thông tin thêm
        if num_features > 0:
            print(f"Kích thước mỗi đặc trưng: {features[0].shape if hasattr(features[0], 'shape') else len(features[0])}")
        else:
            print("Không có đặc trưng nào trong file.")

    except Exception as e:
        print(f"Lỗi khi đọc file .pkl: {e}")

# Đường dẫn tới file .pkl
pkl_file_path = "mediapipe_features.pkl"

# Gọi hàm kiểm tra
check_pkl_file(pkl_file_path)
