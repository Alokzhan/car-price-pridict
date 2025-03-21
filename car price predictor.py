import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt
# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2


# Load dataset
def load_data(file_path=r'D:\pycharm\car price.csv'):
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Columns: {df.columns.tolist()}")
        print(f"Data shape: {df.shape}")

        # Validate required columns
        required_columns = ['Year', 'Brand', 'Mileage', 'EngineSize', 'FuelType', 'Price']
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            print(f"Missing required columns: {missing}")
            print("Falling back to synthetic data")
            return generate_synthetic_data()

        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Using generated data instead")
        return generate_synthetic_data()


# Generate synthetic data if CSV is missing or invalid
def generate_synthetic_data():
    np.random.seed(RANDOM_STATE)
    n_samples = 1000
    brands = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes']

    data = pd.DataFrame({
        'Brand': np.random.choice(brands, n_samples),
        'Year': np.random.randint(2010, 2023, n_samples),
        'Mileage': np.random.randint(5000, 80000, n_samples),
        'EngineSize': np.round(np.random.uniform(1.0, 3.5, n_samples), 1),
        'FuelType': np.random.choice(['Petrol', 'Diesel', 'Hybrid'], n_samples),
        'Price': 15000 * (np.random.randint(2010, 2023, n_samples) - 2000) +
                 5000 * np.round(np.random.uniform(1.0, 3.5, n_samples), 1) -
                 0.2 * np.random.randint(5000, 80000, n_samples) +
                 np.random.choice([0, 3000, 5000], n_samples) +
                 np.random.normal(0, 3000, n_samples)
    })
    return data


# Feature engineering
def create_features(df):
    df = df.copy()
    # Create age feature
    df['Age'] = 2023 - df['Year']
    # Create mileage per year
    df['MileagePerYear'] = df['Mileage'] / df['Age']
    # Create engine size categories
    df['EngineCategory'] = pd.cut(df['EngineSize'],
                                  bins=[0, 1.5, 2.5, 3.5],
                                  labels=['Small', 'Medium', 'Large'])
    return df


# Preprocessing pipeline
def build_preprocessor():
    numeric_features = ['Year', 'Mileage', 'EngineSize', 'Age', 'MileagePerYear']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])

    categorical_features = ['Brand', 'FuelType', 'EngineCategory']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


# Model training
def train_model(X, y):
    preprocessor = build_preprocessor()

    models = {
        'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingRegressor(random_state=RANDOM_STATE),
        'LinearRegression': LinearRegression()
    }

    params = {
        'RandomForest': {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10]},
        'GradientBoosting': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1]},
        'LinearRegression': {}
    }

    best_models = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        grid_search = GridSearchCV(pipeline, params[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)

        best_models[name] = grid_search.best_estimator_
        print(f"{name} best params: {grid_search.best_params_}")

    return best_models


# Evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }

    # Plot feature importance for tree-based models
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importances = model.named_steps['model'].feature_importances_
        indices = np.argsort(importances)[-10:]

        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    return metrics


# Real-time prediction interface
def real_time_predictor(model, data):
    def predict(brand, year, mileage, engine_size, fuel_type):
        input_data = pd.DataFrame([{
            'Brand': brand,
            'Year': year,
            'Mileage': mileage,
            'EngineSize': engine_size,
            'FuelType': fuel_type
        }])

        # Feature engineering
        input_data = create_features(input_data)

        # Predict
        prediction = model.predict(input_data)
        print(f"Predicted Price: ${prediction[0]:.2f}")

        # Update graph
        update_graph(data, input_data, prediction[0])

    return predict


# Update graph with real-time prediction
def update_graph(data, input_data, predicted_price):
    fig = go.Figure()

    # Scatter plot of actual data
    fig.add_trace(go.Scatter(
        x=data['Year'],
        y=data['Price'],
        mode='markers',
        name='Actual Prices',
        marker=dict(color='blue')
    ))

    # Highlight predicted point
    fig.add_trace(go.Scatter(
        x=[input_data['Year'].values[0]],
        y=[predicted_price],
        mode='markers',
        name='Predicted Price',
        marker=dict(color='red', size=15)
    ))

    # Update layout
    fig.update_layout(
        title='Car Price Prediction',
        xaxis_title='Year',
        yaxis_title='Price ($)',
        showlegend=True
    )

    fig.show()


# Main function
def main():
    # Load data
    data = load_data()

    # Feature engineering
    data = create_features(data)

    # Split data
    X = data.drop('Price', axis=1)
    y = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Train models
    models = train_model(X_train, y_train)

    # Evaluate and compare models
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}:")
        results[name] = evaluate_model(model, X_test, y_test)
        for metric, value in results[name].items():
            print(f"{metric}: {value:.4f}")

    # Save best model
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_model = models[best_model_name]
    joblib.dump(best_model, 'best_model.pkl')
    print(f"\nSaved best model ({best_model_name}) as best_model.pkl")

    # Real-time prediction interface
    print("\nLaunching real-time predictor...")
    predictor = real_time_predictor(best_model, data)

    # Interactive widget for real-time prediction
    interact(predictor,
             brand=widgets.Dropdown(options=data['Brand'].unique(), description='Brand'),
             year=widgets.IntSlider(min=2000, max=2023, step=1, value=2020, description='Year'),
             mileage=widgets.IntSlider(min=0, max=200000, step=1000, value=50000, description='Mileage'),
             engine_size=widgets.FloatSlider(min=1.0, max=5.0, step=0.1, value=2.0, description='Engine Size'),
             fuel_type=widgets.Dropdown(options=data['FuelType'].unique(), description='Fuel Type'))


if __name__ == "__main__":
    main()