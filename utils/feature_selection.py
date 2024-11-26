from sklearn.ensemble import RandomForestRegressor

def select_important_features(X, y, num_features=10):
    # Khởi tạo mô hình Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Lấy độ quan trọng của các đặc trưng
    importances = rf.feature_importances_
    indices = importances.argsort()[::-1]  # Sắp xếp theo độ quan trọng giảm dần

    # Chọn top `num_features` đặc trưng quan trọng
    num_features = min(num_features, X.shape[1])  # Đảm bảo không chọn quá số đặc trưng có
    important_features = X.columns[indices[:num_features]]
    
    return important_features

