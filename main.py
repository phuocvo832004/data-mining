from utils.data_processing import load_and_process_data
from utils.feature_selection import select_important_features
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def main():
    # Đọc và tiền xử lý dữ liệu
    data = load_and_process_data('data/healthcare_dataset.csv')
    
    # Xác định các cột đặc trưng (features) và mục tiêu (target)
    X = data.drop(columns=['Billing Amount'])
    y = data['Billing Amount']
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra (80% huấn luyện, 20% kiểm tra)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Lựa chọn các đặc trưng quan trọng (top 10 đặc trưng)
    important_features = select_important_features(X_train, y_train, num_features=10)
    X_train_selected = X_train[important_features]
    X_test_selected = X_test[important_features]

    # Mô hình Cây Quyết Định Hồi Quy (Decision Tree Regressor)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train_selected, y_train)  # Huấn luyện mô hình với các đặc trưng quan trọng
    
    # Dự đoán viện phí cho tập kiểm tra
    y_pred = model.predict(X_test_selected)
    
    # Đánh giá mô hình bằng Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    # Lưu mô hình
    model_path = "models/decision_tree.pkl"
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(model, model_path)
    
    return y_pred

if __name__ == "__main__":
    main()
