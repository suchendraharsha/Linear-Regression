import numpy as np
import pandas as pd
from scipy.stats import skew, stats
from sklearn.model_selection import train_test_split
from scaler import create_and_fit_scaler, transform_data  # Import scaler functions

def preprocess_data(df):
    """Preprocesses the hospital readmissions dataset for Logistic Regression."""

    df['age'] = df['age'].str.replace('[','(',regex=False)

    df['diag_1'] = np.where(df['diag_1'] == 'Other',
                            np.where(df['diag_2'] == 'Other',
                                     df['diag_3'],
                                     df['diag_2']
                                     ),
                            df['diag_1'])

    def replace_with_loop(df, column_name):
        for index, value in df[column_name].items():
            if value == 'yes':
                df.loc[index, column_name] = 1
            elif value == 'no':
                df.loc[index, column_name] = 0
            else:
                df.loc[index, column_name] = -1

        df[column_name] = df[column_name].astype('Int64')
        return df

    def replace_with_values(df, column_name):
        for index, value in df[column_name].items():
            if value == 'high':
                df.loc[index, column_name] = 2
            elif value == 'normal':
                df.loc[index, column_name] = 1
            elif value == 'no':
                df.loc[index, column_name] = 0
            else:
                df.loc[index, column_name] = -1

        df[column_name] = df[column_name].astype('Int64')
        return df

    change_df = pd.DataFrame(df['change'])
    replace_with_loop(change_df, 'change')
    df['change_num'] = change_df

    def replace_with_age(df, column_name):
        for index, value in df[column_name].items():
            if value == '(40-50)':
                df.loc[index, column_name] = 0
            elif value == '(50-60)':
                df.loc[index, column_name] = 1
            elif value == '(60-70)':
                df.loc[index, column_name] = 2
            elif value == '(70-80)':
                df.loc[index, column_name] = 3
            elif value == '(80-90)':
                df.loc[index, column_name] = 4
            elif value == '(90-100)':
                df.loc[index, column_name] = 5
            else:
                df.loc[index, column_name] = -1

        df[column_name] = df[column_name].astype('Int64')
        return df

    age_df = pd.DataFrame(df['age'])
    replace_with_age(age_df, 'age')
    df['age_num'] = age_df

    def replace_diag_diabetics(df):
        diagCols = ['diag_1', 'diag_2', 'diag_3']
        diag = 'Diabetes'
        df['Diabetes_ind'] = (df['diag_1'].str.contains(diag, na=False) |
                                 df['diag_2'].str.contains(diag, na=False) |
                                 df['diag_3'].str.contains(diag, na=False)) * 1
        return df

    df = replace_diag_diabetics(df)

    def replace_diag_Circulatory(df):
        diagCols = ['diag_1', 'diag_2', 'diag_3']
        diag = 'Circulatory'
        df['Circulatory_ind'] = (df['diag_1'].str.contains(diag, na=False) |
                                    df['diag_2'].str.contains(diag, na=False) |
                                    df['diag_3'].str.contains(diag, na=False)) * 1
        return df

    df = replace_diag_Circulatory(df)

    def replace_diag_Injury(df):
        diagCols = ['diag_1', 'diag_2', 'diag_3']
        diag = 'Injury'
        df['Injury_ind'] = (df['diag_1'].str.contains(diag, na=False) |
                               df['diag_2'].str.contains(diag, na=False) |
                               df['diag_3'].str.contains(diag, na=False)) * 1
        return df

    df = replace_diag_Injury(df)

    def replace_diag_Digestive(df):
        diagCols = ['diag_1', 'diag_2', 'diag_3']
        diag = 'Digestive'
        df['Digestive_ind'] = (df['diag_1'].str.contains(diag, na=False) |
                                  df['diag_2'].str.contains(diag, na=False) |
                                  df['diag_3'].str.contains(diag, na=False)) * 1
        return df

    df = replace_diag_Digestive(df)

    def replace_diag_Respiratory(df):
        diagCols = ['diag_1', 'diag_2', 'diag_3']
        diag = 'Respiratory'
        df['Respiratory_ind'] = (df['diag_1'].str.contains(diag, na=False) |
                                     df['diag_2'].str.contains(diag, na=False) |
                                     df['diag_3'].str.contains(diag, na=False)) * 1
        return df

    df = replace_diag_Respiratory(df)

    def replace_diag_Musculoskeletal(df):
        diagCols = ['diag_1', 'diag_2', 'diag_3']
        diag = 'Musculoskeletal'
        df['Musculoskeletal_ind'] = (df['diag_1'].str.contains(diag, na=False) |
                                         df['diag_2'].str.contains(diag, na=False) |
                                         df['diag_3'].str.contains(diag, na=False)) * 1
        return df

    df = replace_diag_Musculoskeletal(df)

    readmitted_df = pd.DataFrame(df['readmitted'])
    replace_with_loop(readmitted_df, 'readmitted')
    df['readmitted_num'] = readmitted_df

    diabetes_med_df = pd.DataFrame(df['diabetes_med'])
    replace_with_loop(diabetes_med_df, 'diabetes_med')
    df['diabetes_med_num'] = diabetes_med_df

    glucose_test_df = pd.DataFrame(df['glucose_test'])
    replace_with_values(glucose_test_df, 'glucose_test')
    df['glucose_test_num'] = glucose_test_df

    A1Ctest_df = pd.DataFrame(df['A1Ctest'])
    replace_with_values(A1Ctest_df, 'A1Ctest')
    df['A1Ctest_num'] = A1Ctest_df

    def create_positive_indicator_column(df, indicator_columns, new_column_name):
        df[new_column_name] = (df[indicator_columns].any(axis=1)).astype(int)
        return df

    indicator_columns = [
        'Diabetes_ind', 'Circulatory_ind', 'Injury_ind', 'Digestive_ind',
        'Respiratory_ind', 'Musculoskeletal_ind'
    ]

    df = create_positive_indicator_column(df, indicator_columns, 'any_positive_indicator')

    quantCols = df.select_dtypes(include=[int, float]).columns
    df_features = df[quantCols]
    processingColumns = ['time_in_hospital', 'n_lab_procedures', 'n_procedures',
                         'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']
    df_processingColumns = df_features.copy()

    for i in processingColumns:
        df_processingColumns[i].replace([np.inf, -np.inf], np.nan, inplace=True)
        df_processingColumns[i] = df_processingColumns[i].fillna(df_processingColumns[i].mean())

    def apply_log_transformation(df, columns):
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if (df[col] <= 0).any():
                    print(f"Warning: Column '{col}' contains non-positive values. Adding 1 before log transformation.")
                    df[col] = np.log1p(df[col])
                else:
                    df[col] = np.log(df[col])
            else:
                print(f"Column '{col}' not found or not numeric. Skipping.")
        return df

    df_skewed = apply_log_transformation(df_processingColumns.copy(), processingColumns)

    def reduce_skewness(df, columns):
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                original_skewness = skew(df[col])
                print(f"Original skewness of '{col}': {original_skewness}")
                if abs(original_skewness) > 1:
                    if original_skewness > 0:
                        try:
                            transformed_data, _ = stats.boxcox(df[col] + 1)
                            df[col] = transformed_data
                            print(f"Box-Cox transformation applied to '{col}'.")
                        except:
                            df[col] = np.sqrt(df[col])
                            print(f"Square root transformation applied to '{col}'.")
                    else:
                        df[col] = df[col]**2
                        print(f"Squared transformation applied to '{col}'.")
                elif abs(original_skewness) > 0.5:
                    if original_skewness > 0:
                        df[col] = np.sqrt(df[col])
                        print(f"Square root transformation applied to '{col}'.")
                    else:
                        df[col] = df[col]**2
                        print(f"Squared transformation applied to '{col}'.")
                elif abs(original_skewness) > 0.25:
                    if original_skewness > 0:
                        df[col] = np.log1p(df[col])
                        print(f"Log1p transformation applied to '{col}'.")
                    else:
                        df[col] = df[col]**2
                        print(f"Squared transformation applied to '{col}'.")
                else:
                    print(f"Skewness of '{col}' is within acceptable range.")
                new_skewness = skew(df[col])
                print(f"New skewness of '{col}': {new_skewness}\n")
            else:
                print(f"Column '{col}' not found or not numeric. Skipping.\n")
        return df

    df_skewed_skewed = reduce_skewness(df_skewed.copy(), processingColumns)
    X = df_skewed_skewed.drop(columns=['readmitted_num', 'Musculoskeletal_ind', 'Respiratory_ind',
                                        'Digestive_ind', 'Injury_ind', 'Circulatory_ind',
                                        'any_positive_indicator'], axis=1)
    y = df_skewed_skewed["readmitted_num"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

    # Use the functions from scaler.py
    scaler = create_and_fit_scaler(X_train)
    X_train_scaled = transform_data(scaler, X_train)
    X_test_scaled = transform_data(scaler, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler