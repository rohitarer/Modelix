from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app)

# Temporary storage for uploaded files
uploaded_files = {}

# Boolean actions template
actions_template = {
    "handle_missing_numeric": False,
    "handle_missing_categorical": False,
    "one_hot_encoding": False,
    "label_encoding_target": False,
    "scaling": False,
    "train_test_split": False,
}


def reset_actions(actions):
    """Reset all actions to False."""
    for key in actions:
        actions[key] = False
    return actions


def preprocess_data(df, x_columns, y_column, actions):
    """Preprocess the dataset based on specified columns."""
    preprocessing_steps = []
    reset_actions(actions)

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number) and df[column].isna().sum() > 0:
            df[column] = imputer.fit_transform(df[[column]])
            actions["handle_missing_numeric"] = True
            preprocessing_steps.append(
                f"df['{column}'] = SimpleImputer(strategy='mean').fit_transform(df[['{column}']])"
            )
        elif df[column].isna().sum() > 0:
            df[column] = df[column].fillna(df[column].mode()[0])
            actions["handle_missing_categorical"] = True
            preprocessing_steps.append(
                f"df['{column}'] = df['{column}'].fillna(df['{column}'].mode()[0])"
            )

    # Separate features (X) and target (y)
    X = df[x_columns]
    y = df[y_column]

    # One-hot encoding for categorical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        ct = ColumnTransformer(
            transformers=[("onehot", OneHotEncoder(drop="first"), categorical_cols)],
            remainder="passthrough",
        )
        X = ct.fit_transform(X)
        actions["one_hot_encoding"] = True
        preprocessing_steps.append(
            f"X = ColumnTransformer(transformers=[('onehot', OneHotEncoder(drop='first'), {list(categorical_cols)})], remainder='passthrough').fit_transform(X)"
        )

    # Label encode target variable if categorical
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
        actions["label_encoding_target"] = True
        preprocessing_steps.append(f"y = LabelEncoder().fit_transform(y)")

    return X, y, preprocessing_steps


@app.route("/")
def index():
    return "Welcome to the ML API!"


@app.route("/get_columns", methods=["POST"])
def get_columns():
    """Retrieve column names from uploaded CSV."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_name = file.filename

    if not file_name.endswith(".csv"):
        return jsonify({"error": "Invalid file type. Only CSV files are supported."}), 400

    try:
        logging.info(f"Reading file: {file_name}")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        file.save(temp_file.name)
        uploaded_files[file_name] = temp_file.name
        df = pd.read_csv(temp_file.name)
        columns = df.columns.tolist()
        logging.info(f"Columns retrieved: {columns}")
        return jsonify({"columns": columns, "file_name": file_name})
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/process_csv", methods=["POST"])
def process_csv():
    """Process the CSV file and train models."""
    try:
        data = request.json
        file_name = data.get("file_name")
        x_columns = data.get("x_columns")
        y_column = data.get("y_column")

        if not file_name or file_name not in uploaded_files:
            return jsonify({"error": "File not found. Please upload a file first."}), 400

        if not x_columns or not y_column:
            return jsonify({"error": "Missing x_columns or y_column"}), 400

        # Load file and validate columns
        df = pd.read_csv(uploaded_files[file_name])
        x_columns = [df.columns[int(col)] if col.isdigit() else col for col in x_columns]
        y_column = df.columns[int(y_column)] if y_column.isdigit() else y_column

        if not set(x_columns).issubset(df.columns) or y_column not in df.columns:
            return jsonify({"error": "Invalid columns specified."}), 400

        # Preprocess data
        actions = actions_template.copy()
        X, y, preprocessing_steps = preprocess_data(df, x_columns, y_column, actions)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        actions["train_test_split"] = True
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        actions["scaling"] = True

        # Train models
        models = {
            "Logistic Regression": LogisticRegression(),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Support Vector Machine (SVM)": SVC(kernel="linear"),
            "Kernel SVM": SVC(kernel="rbf"),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100),
        }

        highest_accuracy = -float("inf")
        best_model_name = None
        best_model_code = ""

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)

                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_model_name = name
                    best_model_code = f"""
# Generated Code for {name}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.{model.__class__.__name__.lower()} import {model.__class__.__name__}

# Load dataset
df = pd.read_csv("{file_name}")

# Preprocessing
{"\\n".join(preprocessing_steps)}

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = {model.__class__.__name__}()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {{accuracy}}")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
"""
            except Exception as e:
                logging.warning(f"Error training model {name}: {e}")

        return jsonify(
            {
                "best_model": best_model_name,
                "accuracy": highest_accuracy,
                "code_template": best_model_code.strip(),
                "actions_performed": actions,
            }
        )
    except Exception as e:
        logging.error(f"Error processing CSV: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
