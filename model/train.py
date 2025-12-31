import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Function to get the clean data
def get_clean_data():
    # Load the data
    data = pd.read_csv("dataset/data.csv")
    
    # Drop the columns that are not required
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    # Map the diagnosis column to 0 and 1
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })

    # Return the clean data
    return data

def create_model(data):
    # Split the data
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    # Return the model and the scaler
    return model, scaler

def main():
    # Get the clean data
    data = get_clean_data()

    # Save the clean data
    data.to_csv('model/data_cleaned.csv', index=False)

    # Create the model
    model, scaler = create_model(data)

    # Save the model and the scaler
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

# Entry point for the script
if __name__ == '__main__':
    main()