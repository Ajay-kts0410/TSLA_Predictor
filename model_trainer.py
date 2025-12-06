import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

def create_model_and_prepare_data(data, look_back=60):
    """
    Prepares data for LSTM and creates/trains the model.
    """
    # Use 'Close_TSLA' and 'Sentiment' as the features for the model
    dataset = data[['Close_TSLA', 'Sentiment']].values
    
    # Normalize the data to a range between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create sequences for the LSTM model
    x, y = [], []
    for i in range(look_back, len(scaled_data)):
        x.append(scaled_data[i-look_back:i, :])
        y.append(scaled_data[i, 0]) # Predict the Close price (the first feature)

    x, y = np.array(x), np.array(y)

    # Split the data into training and testing sets (80% training, 20% testing)
    train_size = int(len(x) * 0.80)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape for LSTM input: [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=1)
    
    # Evaluate the model on the test data
    loss = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nModel Loss on Test Data: {loss:.4f}")

    # Save the trained model and the scaler for future use in the web app
    model.save('tsla_model.h5')
    joblib.dump(scaler, 'scaler.pkl')

    print("Model trained and saved!")

if __name__ == "__main__":
    data = pd.read_csv('TSLA_with_sentiment.csv', index_col='Date', parse_dates=True)
    create_model_and_prepare_data(data)