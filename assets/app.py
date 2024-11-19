from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import tempfile  # Import this module for temporary file handling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Temporary storage for uploaded files
uploaded_files = {}

# Actions tracking template
actions_template = {
    "handle_missing_numeric": False,
    "handle_missing_categorical": False,
    "one_hot_encoding": False,
    "label_encoding_target": False,
    "scaling": False,
    "train_test_split": False,
}

# Reset actions for each model
def reset_actions(actions):
    for key in actions:
        actions[key] = False
    return actions

@app.route('/')
def index():
    return "Welcome to the ML API!"

@app.route('/get_columns', methods=['POST'])
def get_columns():
    """Retrieve column names from the uploaded CSV file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_name = file.filename

    if not file_name.endswith('.csv'):
        return jsonify({"error": "Invalid file type. Only CSV files are supported."}), 400

    try:
        logging.info(f"Reading file: {file_name}")

        # Save file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        file.save(temp_file.name)

        # Store reference for processing
        uploaded_files[file_name] = temp_file.name

        df = pd.read_csv(temp_file.name)
        columns = df.columns.tolist()
        logging.info(f"Columns retrieved: {columns}")
        return jsonify({"columns": columns, "file_name": file_name})
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process_csv', methods=['POST'])
def process_csv():
    """Process CSV file with specified X and Y columns."""
    file_name = request.json.get('file_name')
    x_columns = request.json.get('x_columns')
    y_column = request.json.get('y_column')

    if not file_name or file_name not in uploaded_files:
        return jsonify({"error": "File not found. Please upload a file first."}), 400

    if not x_columns or not y_column:
        return jsonify({"error": "Missing x_columns or y_column"}), 400

    x_columns = x_columns.split(',')
    try:
        # Attempt to convert to integers (indices), otherwise treat as column names
        x_columns = [int(col) if col.isdigit() else col for col in x_columns]
        y_column = int(y_column) if y_column.isdigit() else y_column
    except ValueError:
        return jsonify({"error": "Invalid column indices or names"}), 400

    try:
        file_path = uploaded_files[file_name]
        logging.info(f"Processing file: {file_name}")

        df = pd.read_csv(file_path)

        # If columns are indices, convert them to names
        x_is_indices = isinstance(x_columns[0], int)
        y_is_index = isinstance(y_column, int)
        if x_is_indices:
            x_columns_names = [df.columns[i] for i in x_columns]
        else:
            x_columns_names = x_columns

        if y_is_index:
            y_column_name = df.columns[y_column]
        else:
            y_column_name = y_column

        # Initialize variables
        highest_r2 = -float('inf')
        best_model_name = None
        best_model_code = ""
        best_model_actions = {}
        preprocessing_steps = []

        # Preprocess data
        def preprocess_data():
            """Preprocess the dataset and track actions."""
            reset_actions(actions_template)
            preprocessing = []

            # Handle missing values
            imputer = SimpleImputer(strategy="mean")
            for column in df.columns:
                if df[column].isna().sum() > 0:
                    if np.issubdtype(df[column].dtype, np.number):
                        df[column] = imputer.fit_transform(df[[column]])
                        actions_template["handle_missing_numeric"] = True
                        preprocessing.append(f"df['{column}'] = SimpleImputer(strategy='mean').fit_transform(df[['{column}']])")
                    else:
                        df[column] = df[column].fillna(df[column].mode()[0])
                        actions_template["handle_missing_categorical"] = True
                        preprocessing.append(f"df['{column}'] = df['{column}'].fillna(df['{column}'].mode()[0])")

            # Prepare X and y
            X = df[x_columns_names].values
            y = df[y_column_name].values

            # One-hot encode categorical features in X
            categorical_cols = df[x_columns_names].select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                ct = ColumnTransformer(
                    transformers=[('onehot', OneHotEncoder(drop='first'), categorical_cols)],
                    remainder='passthrough'
                )
                X = ct.fit_transform(X)
                actions_template["one_hot_encoding"] = True
                preprocessing.append(f"X = ColumnTransformer(transformers=[('onehot', OneHotEncoder(drop='first'), {list(categorical_cols)})], remainder='passthrough').fit_transform(X)")

            # Label encode y if it is categorical
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y = le.fit_transform(y)
                actions_template["label_encoding_target"] = True
                preprocessing.append(f"y = LabelEncoder().fit_transform(y)")

            return X, y, preprocessing

        # Preprocess data
        X, y, preprocessing_steps = preprocess_data()

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        actions_template["train_test_split"] = True

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        actions_template["scaling"] = True
        preprocessing_steps.append("scaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)")

        # Train models
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "ElasticNet Regression": ElasticNet(alpha=0.1),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Support Vector Regression": SVR(),
        }

        for name, model in models.items():
            reset_actions(actions_template)  # Reset boolean tracking
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                r2 = r2_score(y_test, predictions)

                if r2 > highest_r2:
                    highest_r2 = r2
                    best_model_name = name
                    best_model_actions = actions_template.copy()
                    best_model_code = f"""
# Generated Code for {name}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import {model.__class__.__name__}

# Load dataset
df = pd.read_csv("{file_name}")

# Selecting features and target
X = df[{x_columns_names}].values
y = df['{y_column_name}'].values

# Preprocessing steps
{"\\n".join(preprocessing_steps)}

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = {model.__class__.__name__}()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"R2 Score: {{r2}}")
"""
            except Exception as e:
                logging.warning(f"Error training model {name}: {e}")

        return jsonify({
            "best_model": best_model_name,
            "r2": highest_r2,
            "code_template": best_model_code.strip(),
            "actions_performed": best_model_actions
        })
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
