import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to Train & Save Model
def train_and_save_model(model_path="model.pkl"):
    """Train Model and Save to File"""

    # Load dataset
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]
    df.dropna(inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Split dataset
    X = df[['Pclass', 'Sex', 'Age', 'Fare']]
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    joblib.dump(model, model_path)
    print(f"✅ Model saved at {model_path} with Accuracy: {accuracy:.2f}")

    return accuracy  # Return accuracy for monitoring
