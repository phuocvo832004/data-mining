import pandas as pd

def load_and_process_data(file_path):
    data = pd.read_csv(file_path)

    columns_to_drop = ['Name', 'Room Number', 'Discharge Date', 'Doctor','Gender', 'Date of Admission', 'Test Results'] 
    data = data.drop(columns=columns_to_drop, axis=1)

    data = data.dropna()

    categorical_cols = ['Blood Type', 'Medical Condition', 
                        'Hospital', 'Insurance Provider', 'Admission Type', 'Medication']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    return data
