from sklearn.preprocessing import StandardScaler
import joblib

def create_and_fit_scaler(X_train):
    """Creates and fits a StandardScaler on the training data.

    Args:
        X_train (pd.DataFrame or np.ndarray): The training features.

    Returns:
        StandardScaler: The fitted StandardScaler object.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def transform_data(scaler, data):
    """Transforms data using a fitted StandardScaler.

    Args:
        scaler (StandardScaler): The fitted StandardScaler object.
        data (pd.DataFrame or np.ndarray): The data to transform.

    Returns:
        np.ndarray: The scaled data.
    """
    return scaler.transform(data)

def save_scaler(scaler, filepath='models/lr_scaler.pkl'):
    """Saves the StandardScaler object to a file.

    Args:
        scaler (StandardScaler): The fitted StandardScaler object.
        filepath (str, optional): The path to save the scaler.
            Defaults to 'models/lr_scaler.pkl'.
    """
    joblib.dump(scaler, filepath)
    print(f"Scaler saved to {filepath}")

def load_scaler(filepath='models/lr_scaler.pkl'):
    """Loads a StandardScaler object from a file.

    Args:
        filepath (str, optional): The path to load the scaler from.
            Defaults to 'models/lr_scaler.pkl'.

    Returns:
        StandardScaler or None: The loaded StandardScaler object, or None if an error occurs.
    """
    try:
        scaler = joblib.load(filepath)
        print(f"Scaler loaded from {filepath}")
        return scaler
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None