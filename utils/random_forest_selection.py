import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def calculate_feature_importance(df, target_column):
    # Loại bỏ hoặc thay thế giá trị NaN trong cột mục tiêu
    if df[target_column].isnull().any():
        print(f"Warning: {target_column} contains NaN values. Filling with mean.")
        df[target_column] = df[target_column].fillna(df[target_column].mean())

    # Tách dữ liệu
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Mã hóa các cột phân loại
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Huấn luyện mô hình
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Lấy độ quan trọng của thuộc tính
    feature_importances = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return feature_importances
