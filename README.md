# TCS-Stock-Data
# TCS Stock Price Prediction with Linear Regression

This project aims to predict the stock prices of Tata Consultancy Services (TCS) using a simple linear regression model.  It utilizes historical stock data to train the model and then makes basic future price predictions.

## Key Features

* **Data Acquisition:** Fetches historical TCS stock data from a CSV file.
* **Data Preprocessing:**
    * Loads stock data and converts the 'Date' column to datetime format.
    * Sorts data by date.
    * Handles missing values by forward filling.
* **Visualization:** Plots the historical close prices of TCS stock.
* **Model Training:**
    * Uses a Linear Regression model for price prediction.
    * Splits the data into training and testing sets.
* **Model Evaluation:** Calculates and prints the Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to assess the model's accuracy.
* **Future Prediction:** Predicts stock prices for the next 30 days using a simple extrapolation of the trained model.
* **Future Visualization:** Combines the historical data with the future predictions in a plot.

##   Dependencies

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

##   Usage

1.  **Data:** Ensure you have the TCS stock data in a CSV file named `TCS_stock_history.csv` in the same directory as the script. The CSV should contain columns like 'Date', 'Open', 'High', 'Low', 'Close', and 'Volume'.
2.  **Libraries:** Install the required Python libraries using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  **Run:** Execute the Python script or Jupyter Notebook to:
    * Fetch and preprocess the data.
    * Train the linear regression model.
    * Evaluate the model's performance.
    * Generate and display future price predictions.

##   Limitations

* **Model Simplicity:** The Linear Regression model used is very basic. Stock prices are influenced by many complex factors, and a simple linear model will likely have limited predictive power.
* **Feature Engineering:** The model uses only the day count as a feature, which is a gross oversimplification.  More sophisticated feature engineering (e.g., using technical indicators, previous day's prices) would be necessary for a more realistic model.
* **Prediction Accuracy:** Due to the limitations above, the future price predictions should not be considered financial advice. They are purely for illustrative purposes.

**Disclaimer:** This project is for educational and demonstration purposes only.  It is not intended to provide financial advice.  Trading in the stock market involves significant risk, and you could lose money.  Always consult with a qualified financial advisor before making any investment decisions.
