import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures

# Boolean tracking for all actions
actions_template = {
    "handle_missing_numeric": False,
    "handle_missing_categorical": False,
    "one_hot_encoding": False,
    "label_encoding_target": False,
    "scaling": False,
    "train_test_split": False,
    "polynomial_features": False,
}

# Function to reset boolean values
def reset_actions(actions):
    for key in actions:
        actions[key] = False
    return actions

# Function to preprocess data
def preprocess_data(df, x_columns, y_column, actions):
    preprocessing_steps = []

    # Handle missing values for numeric and categorical columns
    imputer = SimpleImputer(strategy="mean")
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):  # Numerical columns
            if df[column].isna().sum() > 0:
                df[column] = imputer.fit_transform(df[[column]])
                preprocessing_steps.append(f"df['{column}'] = SimpleImputer(strategy='mean').fit_transform(df[['{column}']])")
                actions["handle_missing_numeric"] = True
        else:  # Categorical columns
            if df[column].isna().sum() > 0:
                df[column] = df[column].fillna(df[column].mode()[0])
                preprocessing_steps.append(f"df['{column}'] = df['{column}'].fillna(df['{column}'].mode()[0])")
                actions["handle_missing_categorical"] = True
    
    # Separate features (X) and target (y)
    X = df[x_columns]
    y = df[y_column]

    # Apply encoding to categorical columns in X
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        ct = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(drop='first'), categorical_cols)
            ],
            remainder='passthrough'
        )
        X = ct.fit_transform(X)
        preprocessing_steps.append(f"X = ColumnTransformer(transformers=[('onehot', OneHotEncoder(drop='first'), {list(categorical_cols)})], remainder='passthrough').fit_transform(X)")
        actions["one_hot_encoding"] = True

    # Label encode the target variable if categorical
    if np.issubdtype(y.dtype, np.object_):
        le = LabelEncoder()
        y = le.fit_transform(y)
        preprocessing_steps.append(f"y = LabelEncoder().fit_transform(y)")
        actions["label_encoding_target"] = True
    else:
        y = y.values

    return X, y, preprocessing_steps

# Function to split data
def split_data(X, y, actions):
    actions["scaling"] = True
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    actions["train_test_split"] = True
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

# Function to train and evaluate models
def train_models(X_train, X_test, y_train, y_test):
    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "ElasticNet Regression": ElasticNet(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Support Vector Regression": SVR(),
    }

    # Initialize tracking variables
    highest_r2 = -float('inf')
    best_model_name = None
    best_model_code = ""
    best_model_actions = {}

    # Train and evaluate models
    for name, model in models.items():
        actions = reset_actions(actions_template.copy())  # Reset boolean tracking for each model

        if name == "Polynomial Regression":
            poly = PolynomialFeatures(degree=2)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)
            model.fit(X_train_poly, y_train)
            predictions = model.predict(X_test_poly)
            actions["polynomial_features"] = True
            model_code = (
                "from sklearn.preprocessing import PolynomialFeatures\n"
                "poly = PolynomialFeatures(degree=2)\n"
                "X_train_poly = poly.fit_transform(X_train)\n"
                "X_test_poly = poly.transform(X_test)\n"
                f"model = {model.__class__.__name__}()\n"
                "model.fit(X_train_poly, y_train)\n"
            )
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            model_code = f"model = {model.__class__.__name__}({', '.join([f'{k}={repr(v)}' for k, v in model.get_params().items()])})\n"
            model_code += "model.fit(X_train, y_train)\n"

        # Evaluate model
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Track the best model
        if r2 > highest_r2:
            highest_r2 = r2
            best_model_name = name
            best_model_code = model_code
            best_model_actions = actions.copy()

    return best_model_name, highest_r2, best_model_code, best_model_actions

# Generate the template dynamically based on actions
def generate_code_template(file_path, preprocessing_steps, best_model_code, actions):
    code_template = []

    # Add imports based on actions
    code_template.append("import pandas as pd")
    if actions["handle_missing_numeric"] or actions["handle_missing_categorical"]:
        code_template.append("from sklearn.impute import SimpleImputer")
    if actions["one_hot_encoding"]:
        code_template.append("from sklearn.preprocessing import OneHotEncoder")
        code_template.append("from sklearn.compose import ColumnTransformer")
    if actions["label_encoding_target"]:
        code_template.append("from sklearn.preprocessing import LabelEncoder")
    code_template.append("from sklearn.model_selection import train_test_split")
    if actions["scaling"]:
        code_template.append("from sklearn.preprocessing import StandardScaler")
    code_template.append("from sklearn.metrics import mean_squared_error, r2_score")
    if actions["polynomial_features"]:
        code_template.append("from sklearn.preprocessing import PolynomialFeatures")

    # Load dataset
    code_template.append(f"\ndf = pd.read_csv('{file_path}')")

    # Preprocessing steps
    code_template.extend(preprocessing_steps)

    # Splitting data
    if actions["scaling"]:
        code_template.append("scaler = StandardScaler()")
        code_template.append("X_scaled = scaler.fit_transform(X)")
    if actions["train_test_split"]:
        code_template.append("X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)")

    # Model training and evaluation
    code_template.append("\n# Model training and evaluation")
    code_template.append(best_model_code)
    code_template.append("predictions = model.predict(X_test)")
    code_template.append("mse = mean_squared_error(y_test, predictions)")
    code_template.append("r2 = r2_score(y_test, predictions)")
    code_template.append("print(f'MSE: {mse}, R2 Score: {r2}')")

    return "\n".join(code_template)

# Main function
if __name__ == "__main__":
    file_path = input("Enter the path to your CSV file: ")
    df = pd.read_csv(file_path)

    print("Columns in the dataset:")
    print(df.columns.tolist())

    x_input = input("Enter the column names or indices for features (X) separated by commas: ")
    y_input = input("Enter the column name or index for the target (Y): ")

    if x_input.replace(',', '').isdigit():
        x_columns = [df.columns[int(i)] for i in x_input.split(',')]
    else:
        x_columns = x_input.split(',')

    if y_input.isdigit():
        y_column = df.columns[int(y_input)]
    else:
        y_column = y_input

    X, y, preprocessing_steps = preprocess_data(df, x_columns, y_column, actions_template)
    X_train, X_test, y_train, y_test, scaler = split_data(X, y, actions_template)
    best_model_name, highest_r2, best_model_code, best_model_actions = train_models(X_train, X_test, y_train, y_test)

    print(f"\nBest Model: {best_model_name}")
    print(f"Best Model R2 Score: {highest_r2}")

    # Generate and display the template
    template = generate_code_template(file_path, preprocessing_steps, best_model_code, best_model_actions)
    print("\n### Complete Python Code ###\n")
    print(template)
