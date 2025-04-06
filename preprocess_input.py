import numpy as np
import pandas as pd
from scaler import transform_data  # Import the transform function

def preprocess_input_lr(data, scaler):
    """Preprocesses input data for Logistic Regression prediction,
    following the same steps as in the training preprocessing.
    """
    df = pd.DataFrame([data])
    if 'age' in df.columns:
        df['age'] = df['age'].str.replace('[','(',regex=False)

    def replace_with_loop_input(df, column_name):
        if column_name in df.columns:
            df[column_name] = df[column_name].replace({'yes': 1, 'no': 0, 'No': 0, 'Yes': 1, 'Ch': 1, 'Steady': 0})
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(-1).astype('Int64')
        return df

    def replace_with_values_input(df, column_name):
        if column_name in df.columns:
            df[column_name] = df[column_name].replace({'high': 2, 'normal': 1, 'no': 0, 'Norm': 1, '>200': 2, '>300': 2, '>7': 2, '>8': 2, 'NaN': 0})
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(-1).astype('Int64')
        return df

    df = replace_with_loop_input(df.copy(), 'change')
    df = replace_with_loop_input(df, 'diabetes_med')

    def replace_with_age_input(df, column_name):
        if column_name in df.columns:
            df[column_name] = df[column_name].replace({'(40-50)': 0, '(50-60)': 1, '(60-70)': 2,
                                                       '(70-80)': 3, '(80-90)': 4, '(90-100)': 5,
                                                       '[40-50)': 0, '[50-60)': 1, '[60-70)': 2,
                                                       '[70-80)': 3, '[80-90)': 4, '[90-100)': 5,
                                                       '[0-10)': -1, '[10-20)': -1, '[20-30)': -1, '[30-40)': -1}) # Handle potential other age groups as -1
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(-1).astype('Int64')
        return df

    df = replace_with_age_input(df.copy(), 'age')
    df = replace_with_values_input(df, 'glucose_test')
    df = replace_with_values_input(df, 'A1Ctest')

    # Replicate the 'Other' diagnosis replacement
    df['diag_1'] = np.where(df['diag_1'] == 'Other',
                            np.where(df['diag_2'] == 'Other',
                                     df['diag_3'],
                                     df['diag_2']
                                     ),
                            df['diag_1'])

    def replace_diag_indicator_input(df, diag, column_prefix):
        diag_cols = ['diag_1', 'diag_2', 'diag_3']
        for i, col in enumerate(diag_cols):
            if col in df.columns:
                df[f'{column_prefix}_ind'] = df.get(col, '').str.contains(diag, na=False).astype(int)
            else:
                df[f'{column_prefix}_ind'] = 0
            # Break after the first match to align with the training logic
            if f'{column_prefix}_ind' in df.columns and df[f'{column_prefix}_ind'].any():
                break
        return df

    df = replace_diag_indicator_input(df, 'Diabetes', 'Diabetes')
    df = replace_diag_indicator_input(df, 'Circulatory', 'Circulatory')
    df = replace_diag_indicator_input(df, 'Injury', 'Injury')
    df = replace_diag_indicator_input(df, 'Digestive', 'Digestive')
    df = replace_diag_indicator_input(df, 'Respiratory', 'Respiratory')
    df = replace_diag_indicator_input(df, 'Musculoskeletal', 'Musculoskeletal')

    indicator_columns = [
        'Diabetes_ind', 'Circulatory_ind', 'Injury_ind', 'Digestive_ind',
        'Respiratory_ind', 'Musculoskeletal_ind'
    ]
    df['any_positive_indicator'] = df[indicator_columns].any(axis=1).astype(int)

    numerical_cols = ['time_in_hospital', 'n_lab_procedures', 'n_procedures',
                      'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency',
                      'change_num', 'age_num',
                      'diabetes_med_num', 'glucose_test_num', 'A1Ctest_num',
                      'any_positive_indicator',
                      'Diabetes_ind', 'Circulatory_ind', 'Injury_ind', 'Digestive_ind',
                      'Respiratory_ind', 'Musculoskeletal_ind'
                      ]

    # Select only the numerical columns present in the input data
    input_data = df[[col for col in numerical_cols if col in df.columns]].fillna(df[[col for col in numerical_cols if col in df.columns]].mean())

    # Ensure the order of columns matches the training data (after dropping in preprocessing.py)
    expected_cols = ['time_in_hospital', 'n_lab_procedures', 'n_procedures',
                     'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency',
                     'change_num', 'age_num','Diabetes_ind',
                     'diabetes_med_num', 'glucose_test_num', 'A1Ctest_num']

    # Add missing columns with 0 if the input doesn't have them
    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[expected_cols] # Ensure correct column order

    scaled_data = transform_data(scaler, input_data)
    return scaled_data