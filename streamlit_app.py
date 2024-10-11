import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math

# Recreate the model architecture
def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(units=256, return_sequences=True),
        keras.layers.LSTM(units=256, return_sequences=False),
        keras.layers.Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load the trained weights
@st.cache_resource
def load_model():
    model = create_model((5, 1))  # Assuming input shape is (5, 1)
    model.load_weights('models/lstm_model_weights.h5')
    return model

# Prepare data function (your original function)
def prepare_data(uploaded_file) :
    data = pd.read_csv(uploaded_file, index_col=1)
    data = data[['Q-P1']]
    data['Cumulative_Sum'] = data['Q-P1'].cumsum()
    data.head()
    data = data[['Cumulative_Sum']]

    # Setting 80 percent data for training
    training_data_len = math.ceil(len(data) * .8)
    training_data_len

    #Splitting the dataset
    train_data = data[:training_data_len].iloc[:,:1]
    test_data = data[training_data_len:].iloc[:,:1]
    print(train_data.shape, test_data.shape)

    # Selecting Open Price values
    dataset_train = train_data.Cumulative_Sum.values
    # Reshaping 1D to 2D array
    dataset_train = np.reshape(dataset_train, (-1,1))

    scaler_train = MinMaxScaler(feature_range=(0,1))
    scaler_test = MinMaxScaler(feature_range=(0,1))
    # scaling dataset
    scaled_train = scaler_train.fit_transform(dataset_train)

    # Selecting Open Price values
    dataset_test = test_data.Cumulative_Sum.values
    # Reshaping 1D to 2D array
    dataset_test = np.reshape(dataset_test, (-1,1))
    # Normalizing values between 0 and 1
    scaled_test = scaler_test.fit_transform(dataset_test)

    return scaled_train, scaled_test, scaler_test, test_data

def split_data(scaled_train, scaled_test, prev=5):
    X_train = []
    y_train = []
    for i in range(prev, len(scaled_train)):
        X_train.append(scaled_train[i-prev:i, 0])
        y_train.append(scaled_train[i, 0])
    
    X_test = []
    y_test = []
    for i in range(prev, len(scaled_test)):
        X_test.append(scaled_test[i-prev:i, 0])
        y_test.append(scaled_test[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    #Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
    y_train = np.reshape(y_train, (y_train.shape[0],1))

    # The data is converted to numpy array
    X_test, y_test = np.array(X_test), np.array(y_test)

    #Reshaping
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    y_test = np.reshape(y_test, (y_test.shape[0],1))

    return X_train, y_train, X_test, y_test

def predict(X_test, regressor, scaler_test, test_data, prev=5):
    y_LSTM = regressor.predict(X_test)
    y_LSTM_O = scaler_test.inverse_transform(y_LSTM)

    y_LSTM_O_reshaped = y_LSTM_O.reshape(-1,)
    predictions = pd.DataFrame({'actual': test_data.Cumulative_Sum[prev:], 'predicted': y_LSTM_O_reshaped})
    predictions['actual_sales'] = predictions['actual'] - predictions['actual'].shift(-1)
    predictions['predicted_sales'] = predictions['predicted'] - predictions['predicted'].shift(-1)

    return predictions, y_LSTM_O_reshaped

# Streamlit app
def main():
    st.title('Sales Prediction App')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Prepare data
        scaled_train, scaled_test, scaler_test, test_data = prepare_data(uploaded_file)

        # Load model
        regressor = load_model()

        X_train, y_train, X_test, y_test = split_data(scaled_train, scaled_test)

        predictions, y_LSTM_O_reshaped = predict(X_test, regressor, scaler_test, test_data)

        # Plot 3
        monthly_test = []
        monthly_preds = []
        each_month_test = []
        each_month_preds = []
        fig, ax = plt.subplots(figsize =(10,6))
        for i in range(1, len(y_LSTM_O_reshaped)):
            if i % 30 == 0:
                monthly_test.append(np.mean(np.array(each_month_test)))
                monthly_preds.append(np.mean(np.array(each_month_preds)))
                each_month_test = []
                each_month_preds = []
            else:
                each_month_test.append(predictions['actual_sales'].iloc[i])
                each_month_preds.append(predictions['predicted_sales'].iloc[i])
        monthly_test = np.array(monthly_test)
        monthly_preds = np.array(monthly_preds)
        ax.plot(monthly_test, label = "actual_sales", color = "g")
        ax.plot(monthly_preds, label = "predicted_sales", color = "brown")
        ax.legend()
        ax.set_title("Sales Prediction (Monthly Average)")
        st.pyplot(fig)

        # Display raw data
        st.subheader('Raw Data')
        st.write(predictions)

if __name__ == '__main__':
    main()