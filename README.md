# Gold Price Forecasting 📈💰

This project uses time series analysis and machine learning techniques to forecast gold prices. By analyzing historical gold price data, we can uncover trends, seasonality, and build predictive models (like ARIMA) to forecast future values.

## Libraries Used 📚

We use several powerful Python libraries for data manipulation and visualization:
- **Pandas**: Data loading and manipulation
- **Matplotlib/Seaborn**: Visualization of trends and patterns
- **Statsmodels**: Statistical models and time series analysis (ARIMA, ACF/PACF)
- **Scikit-learn**: Evaluation metrics for model performance

## Data Overview 📊

The dataset consists of historical gold price data. The date is recorded along with the gold price, and we perform a detailed analysis to understand the dynamics of this time series.

### Insights from Data Visualization 🔍
- **Gold Price Trends**: The gold price shows a clear upward trend over time, suggesting that gold prices have generally increased over the period.
- **Volatility**: There are significant fluctuations in the gold price, indicating high volatility, which is typical for commodity markets.

### Seasonality & Patterns 🌱
- We explored **seasonal decomposition** to separate the trend, seasonal components, and residuals.
- The **trend** shows an increasing pattern, while the **seasonal component** indicates recurring behavior over time.
- The **residuals** are mostly random, suggesting that the model captures most of the structure in the data.

### Stationarity Test 🕵️‍♂️
- The Augmented Dickey-Fuller (ADF) test was applied to check for stationarity. The high p-value suggests that the gold price time series is **non-stationary**, meaning the data has trends or patterns that change over time. This is important for forecasting models like ARIMA.

## Seasonality Analysis 🌀

Using Fast Fourier Transform (FFT), we explored the dominant frequencies in the gold price data. However, the seasonality is somewhat overshadowed by the overall upward trend, meaning the periodic fluctuations are less pronounced than the long-term trend.

## Model Building 🏗️

### Linear Regression Model 📉

A **Linear Regression** model was built to predict the future gold price based on the actual price. This is a simple model where:
- **Training set**: 90% of the data
- **Testing set**: Remaining 10%

The model achieved an **MAE (Mean Absolute Error)** of **0.0228**, **MSE (Mean Squared Error)** of **0.00086**, and a **MAPE (Mean Absolute Percentage Error)** of **3.16%**.

#### Key Interpretation:
- The **MAPE** value of **3.16%** indicates that on average, the model's predictions are **96.84% accurate**.
- The **MSE** and **MAE** values are low, suggesting that the linear regression model performs well in predicting future gold prices.

### ARIMA Model 🧠

We used **ARIMA (AutoRegressive Integrated Moving Average)**, a popular method for time series forecasting. The ARIMA model was trained on the historical data and then used to forecast future gold prices.

#### Key Insights:
- The **ARIMA model** helped us account for trends and autocorrelations in the gold price time series.
- After fine-tuning the model using the **AIC (Akaike Information Criterion)**, the optimal ARIMA order was selected as **(3,1,3)**.
- The forecast from ARIMA showed promising results, with a **low MAPE** indicating good prediction accuracy.

### Model Performance 📊

- The **ARIMA model** performed well with a **low MSE** and **MAPE**, suggesting its effectiveness in forecasting future gold prices. 
- A comparison of the **actual vs predicted** gold prices over the test period demonstrated that the model can capture the overall trend of gold price movements.

## Conclusion 🎉

### Key Takeaways:
- **Trends**: The gold price shows a clear upward trend over time, and this was captured well by both the Linear Regression and ARIMA models.
- **Volatility**: Despite the overall trend, the gold price is highly volatile, and this adds complexity to predictions.
- **ARIMA vs Linear Regression**: While **Linear Regression** is good for a simple prediction, **ARIMA** is better suited for capturing the time-dependent structure (auto-correlation) of the data.
- **Accuracy**: Both models performed well, but ARIMA slightly outperformed Linear Regression in terms of prediction accuracy.

### Future Work 🚀:
- **Model Refinement**: We can further tune the ARIMA model by exploring different combinations of parameters to improve accuracy.
- **Incorporating Exogenous Variables**: Future work could involve including other factors (such as global economic indicators) to improve the forecasts.
- **Advanced Models**: Machine learning models like **Random Forests** or **XGBoost** could be explored for more complex patterns in the data.

---

By leveraging these forecasting techniques, we gain valuable insights into the behavior of gold prices, which can be used to make informed decisions in trading, investment, and economic forecasting. 📊💡
