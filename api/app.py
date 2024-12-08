from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import logging
import tempfile
import traceback

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app)

# Temporary storage for uploaded files
uploaded_files = {}

@app.route('/')
def index():
    return "Welcome to the Combined Regression and Classification ML API!"

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
        # Save the file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        file.save(temp_file.name)
        uploaded_files[file_name] = temp_file.name

        # Read the file to retrieve column names
        df = pd.read_csv(temp_file.name)
        columns = df.columns.tolist()
        logging.info(f"Columns retrieved: {columns}")
        return jsonify({"columns": columns, "file_name": file_name})
    except Exception as e:
        logging.error(f"Error reading CSV file: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/process_csv', methods=['POST'])
def process_csv():
    """Process CSV file for Regression or Classification based on mode."""
    try:
        data = request.json
        logging.info(f"Received payload: {data}")

        # Extract mode (1 = Regression, 2 = Classification), file name, X columns, and Y column
        mode = data.get('mode')  # 1 for Regression, 2 for Classification
        file_name = data.get('file_name')
        x_columns = data.get('x_columns')
        y_column = data.get('y_column')

        if not file_name or file_name not in uploaded_files:
            return jsonify({"error": "File not found. Please upload a file first."}), 400

        if not x_columns or not y_column:
            return jsonify({"error": "Missing x_columns or y_column"}), 400

        # Load the uploaded file
        file_path = uploaded_files[file_name]
        df = pd.read_csv(file_path)
        logging.info(f"Loaded file: {file_name}")

        # Handle column indices or names
        try:
            x_columns = [int(col) if col.isdigit() else col for col in x_columns]
            y_column = int(y_column) if y_column.isdigit() else y_column
        except ValueError:
            return jsonify({"error": "Invalid column indices or names"}), 400

        # Convert indices to column names
        if isinstance(x_columns[0], int):
            x_columns = [df.columns[i] for i in x_columns]
        if isinstance(y_column, int):
            y_column = df.columns[y_column]

        # Validate the columns
        if not set(x_columns).issubset(df.columns):
            return jsonify({"error": f"Invalid x_columns: {x_columns}"}), 400
        if y_column not in df.columns:
            return jsonify({"error": f"Invalid y_column: {y_column}"}), 400

        logging.info(f"Processing columns: X={x_columns}, Y={y_column}")

        # Data preprocessing
        df = df.ffill()
        X = df[x_columns]
        y = df[y_column]

        # Handle categorical columns in X
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            ct = ColumnTransformer(
                transformers=[('onehot', OneHotEncoder(drop='first'), categorical_columns)],
                remainder='passthrough'
            )
            X = ct.fit_transform(X)

        # Encode target column if necessary (for Classification)
        if mode == 2 and not np.issubdtype(y.dtype, np.number):
            y = LabelEncoder().fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Check if X_train is sparse
        if hasattr(X_train, "toarray"):  # Detect sparse matrix
            scaler = StandardScaler(with_mean=False)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Module mappings
        regression_module_map = {
            "LinearRegression": "linear_model",
            "Decision Tree Regressor": "tree",
            "Random Forest Regressor": "ensemble",
            "Support Vector Regressor": "svm"
        }

        classification_module_map = {
            "Logistic Regression": "linear_model",
            "K-Nearest Neighbors": "neighbors",
            "Support Vector Machine": "svm",
            "Naive Bayes": "naive_bayes",
            "Decision Tree": "tree",
            "Random Forest": "ensemble"
        }

        # Process based on mode
        if mode == 1:  # Regression
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "SVR": SVR(),
            }
            best_model_name = None
            best_r2 = -float("inf")

            for name, model in models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                r2 = r2_score(y_test, predictions)
                if r2 > best_r2:
                    best_model_name = name
                    best_r2 = r2

            module_name = regression_module_map.get(best_model_name, "linear_model")
            generated_code = f"""
# Generated Code for {best_model_name}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.{module_name} import {best_model_name}

# Load dataset
df = pd.read_csv("{file_name}")

# Preprocessing
df = df.ffill()
X = df[{x_columns}]
y = df['{y_column}']

categorical_columns = X.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    ct = ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(drop='first'), categorical_columns)],
        remainder='passthrough'
    )
    X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().transform(X_test)

model = {best_model_name}()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
print(f"R2 Score: {{r2}}")
"""
            return jsonify({
                "best_model": best_model_name,
                "r2": best_r2,
                "code_template": generated_code.strip()
            })

        elif mode == 2:  # Classification
            models = {
                "LogisticRegression": LogisticRegression(),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "SVC": SVC(),
                "GaussianNB": GaussianNB(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
            }
            best_model_name = None
            best_accuracy = 0

            # Convert sparse to dense if necessary
            if hasattr(X_train, "toarray"):  # Detect sparse matrix
                X_train = X_train.toarray()
                X_test = X_test.toarray()

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    accuracy = accuracy_score(y_test, model.predict(X_test))
                    if accuracy > best_accuracy:
                        best_model_name = name
                        best_accuracy = accuracy
                except Exception as e:
                    logging.warning(f"Model {name} failed: {e}")

            module_name = classification_module_map.get(best_model_name, "linear_model")
            generated_code = f"""
# Generated Code for {best_model_name}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.{module_name} import {best_model_name}

# Load dataset
df = pd.read_csv("{file_name}")

# Preprocessing
df = df.ffill()
X = df[{x_columns}]
y = df['{y_column}']

categorical_columns = X.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    ct = ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(drop='first'), categorical_columns)],
        remainder='passthrough'
    )
    X = ct.fit_transform(X)

if not np.issubdtype(y.dtype, np.number):
    y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if hasattr(X_train, "toarray"):  # Detect sparse matrix
    X_train = X_train.toarray()
    X_test = X_test.toarray()

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().transform(X_test)

model = {best_model_name}()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {{accuracy}}")
"""
            return jsonify({
                "best_model": best_model_name,
                "accuracy": best_accuracy,
                "code_template": generated_code.strip()
            })

        else:
            return jsonify({"error": "Invalid mode. Use 1 for Regression and 2 for Classification."}), 400

    except Exception as e:
        logging.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# @app.route('/process_csv', methods=['POST'])
# def process_csv():
#     """Process CSV file for Regression or Classification based on mode."""
#     try:
#         data = request.json
#         logging.info(f"Received payload: {data}")

#         # Extract mode (1 = Regression, 2 = Classification), file name, X columns, and Y column
#         mode = data.get('mode')  # 1 for Regression, 2 for Classification
#         file_name = data.get('file_name')
#         x_columns = data.get('x_columns')
#         y_column = data.get('y_column')

#         if not file_name or file_name not in uploaded_files:
#             return jsonify({"error": "File not found. Please upload a file first."}), 400

#         if not x_columns or not y_column:
#             return jsonify({"error": "Missing x_columns or y_column"}), 400

#         # Load the uploaded file
#         file_path = uploaded_files[file_name]
#         df = pd.read_csv(file_path)
#         logging.info(f"Loaded file: {file_name}")

#         # Handle column indices or names
#         try:
#             x_columns = [int(col) if col.isdigit() else col for col in x_columns]
#             y_column = int(y_column) if y_column.isdigit() else y_column
#         except ValueError:
#             return jsonify({"error": "Invalid column indices or names"}), 400

#         # Convert indices to column names
#         if isinstance(x_columns[0], int):
#             x_columns = [df.columns[i] for i in x_columns]
#         if isinstance(y_column, int):
#             y_column = df.columns[y_column]

#         # Validate the columns
#         if not set(x_columns).issubset(df.columns):
#             return jsonify({"error": f"Invalid x_columns: {x_columns}"}), 400
#         if y_column not in df.columns:
#             return jsonify({"error": f"Invalid y_column: {y_column}"}), 400

#         logging.info(f"Processing columns: X={x_columns}, Y={y_column}")

#         # Data preprocessing
#         df = df.ffill()
#         X = df[x_columns]
#         y = df[y_column]

#         # Handle categorical columns in X
#         categorical_columns = X.select_dtypes(include=['object']).columns
#         if len(categorical_columns) > 0:
#             ct = ColumnTransformer(
#                 transformers=[('onehot', OneHotEncoder(drop='first'), categorical_columns)],
#                 remainder='passthrough'
#             )
#             X = ct.fit_transform(X)

#         # Encode target column if necessary (for Classification)
#         if mode == 2 and not np.issubdtype(y.dtype, np.number):
#             y = LabelEncoder().fit_transform(y)

#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Check if X_train is sparse
#         if hasattr(X_train, "toarray"):  # Detect sparse matrix
#             scaler = StandardScaler(with_mean=False)
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)
#         else:
#             scaler = StandardScaler()
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)

#         # Process based on mode
#         if mode == 1:  # Regression
#             models = {
#                 "LinearRegression": LinearRegression(),
#                 "Decision Tree Regressor": DecisionTreeRegressor(),
#                 "Random Forest Regressor": RandomForestRegressor(),
#                 "Support Vector Regressor": SVR(),
#             }
#             best_model_name = None
#             best_r2 = -float("inf")

#             for name, model in models.items():
#                 model.fit(X_train, y_train)
#                 predictions = model.predict(X_test)
#                 r2 = r2_score(y_test, predictions)
#                 if r2 > best_r2:
#                     best_model_name = name
#                     best_r2 = r2

#             generated_code = f"""
# # Generated Code for {best_model_name}
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import r2_score
# from sklearn.{best_model_name.replace(' ', '')} import {best_model_name}

# # Load dataset
# df = pd.read_csv("{file_name}")

# # Preprocessing
# df = df.ffill()
# X = df[{x_columns}]
# y = df['{y_column}']

# categorical_columns = X.select_dtypes(include=['object']).columns
# if len(categorical_columns) > 0:
#     ct = ColumnTransformer(
#         transformers=[('onehot', OneHotEncoder(drop='first'), categorical_columns)],
#         remainder='passthrough'
#     )
#     X = ct.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train = StandardScaler().fit_transform(X_train)
# X_test = StandardScaler().transform(X_test)

# model = {best_model_name}()
# model.fit(X_train, y_train)

# predictions = model.predict(X_test)
# r2 = r2_score(y_test, predictions)
# print(f"R2 Score: {{r2}}")
# """
#             return jsonify({
#                 "best_model": best_model_name,
#                 "r2": best_r2,
#                 "code_template": generated_code.strip()
#             })

#         elif mode == 2:  # Classification
#             models = {
#                 "Logistic Regression": LogisticRegression(),
#                 "K-Nearest Neighbors": KNeighborsClassifier(),
#                 "Support Vector Machine": SVC(),
#                 "Naive Bayes": GaussianNB(),
#                 "Decision Tree": DecisionTreeClassifier(),
#                 "Random Forest": RandomForestClassifier(),
#             }
#             best_model_name = None
#             best_accuracy = 0

#             # Convert sparse to dense if necessary
#             if hasattr(X_train, "toarray"):  # Detect sparse matrix
#                 X_train = X_train.toarray()
#                 X_test = X_test.toarray()

#             for name, model in models.items():
#                 try:
#                     model.fit(X_train, y_train)
#                     accuracy = accuracy_score(y_test, model.predict(X_test))
#                     if accuracy > best_accuracy:
#                         best_model_name = name
#                         best_accuracy = accuracy
#                 except Exception as e:
#                     logging.warning(f"Model {name} failed: {e}")

#             generated_code = f"""
# # Generated Code for {best_model_name}
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import accuracy_score
# from sklearn.{best_model_name.replace(' ', '')} import {best_model_name}

# # Load dataset
# df = pd.read_csv("{file_name}")

# # Preprocessing
# df = df.ffill()
# X = df[{x_columns}]
# y = df['{y_column}']

# categorical_columns = X.select_dtypes(include=['object']).columns
# if len(categorical_columns) > 0:
#     ct = ColumnTransformer(
#         transformers=[('onehot', OneHotEncoder(drop='first'), categorical_columns)],
#         remainder='passthrough'
#     )
#     X = ct.fit_transform(X)

# if not np.issubdtype(y.dtype, np.number):
#     y = LabelEncoder().fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# if hasattr(X_train, "toarray"):  # Detect sparse matrix
#     X_train = X_train.toarray()
#     X_test = X_test.toarray()

# X_train = StandardScaler().fit_transform(X_train)
# X_test = StandardScaler().transform(X_test)

# model = {best_model_name}()
# model.fit(X_train, y_train)

# predictions = model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy: {{accuracy}}")
# """
#             return jsonify({
#                 "best_model": best_model_name,
#                 "accuracy": best_accuracy,
#                 "code_template": generated_code.strip()
#             })


#         else:
#             return jsonify({"error": "Invalid mode. Use 1 for Regression and 2 for Classification."}), 400

#     except Exception as e:
#         logging.error(f"Unexpected error: {traceback.format_exc()}")
#         return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)




