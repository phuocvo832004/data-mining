from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib  # Dùng để lưu mô hình

def train_decision_tree(X_train, y_train, model_path="models/decision_tree.pkl"):
    # Huấn luyện Decision Tree Regressor
    dt = DecisionTreeRegressor( random_state=42, n_jobs=-1)
    dt.fit(X_train, y_train)

    # Lưu mô hình
    joblib.dump(dt, model_path)
    return dt

def evaluate_model(model, X_test, y_test):
    # Dự đoán
    y_pred = model.predict(X_test)

    # Tính toán Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_pred
