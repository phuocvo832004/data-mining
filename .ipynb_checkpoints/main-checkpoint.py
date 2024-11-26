from preprocess.data_loader import load_data
from preprocess.preprocess_data import preprocess_data
from models.decision_tree import train_decision_tree
from models.evaluate import evaluate_model

def main():
    # Bước 1: Load dữ liệu
    df = load_data("./data/healthcare_dataset.csv")

    # Bước 2: Tiền xử lý dữ liệu
    df = preprocess_data(df)

    # Bước 3: Huấn luyện mô hình cây quyết định
    target_column = 'Billing Amount'
    model, X_test, y_test = train_decision_tree(df, target_column)

    # Bước 4: Đánh giá mô hình
    results = evaluate_model(model, X_test, y_test)
    print("Evaluation Results:", results)

if __name__ == "__main__":
    main()
