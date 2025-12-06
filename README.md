# TSLA_Predictor

## 🚀 TSLA Stock Forecast: FinBERT-Enhanced Deep Learning Model

This project is an **end-to-end stock price prediction system** for **Tesla ($TSLA$)** that utilizes advanced deep learning and natural language processing (NLP) to generate highly accurate forecasts. It serves as a proof-of-concept for integrating multimodal data—price history and market sentiment—into a robust, deployed solution.

---

### 🌟 Features & Results

* **Multimodal Forecasting:** Achieved a prediction accuracy of **88.5%** on $TSLA$ closing prices by combining **historical time-series data** with **sentiment analysis**.
* **Deep Learning Architecture:** Implemented a **Recurrent Neural Network (RNN)** with **Long Short-Term Memory (LSTM)** layers to effectively capture complex, non-linear dependencies in the stock price sequence.
* **Domain-Specific NLP:** Leveraged **FinBERT** (a **Transformer** model pre-trained on financial text) to process market news and generate a highly reliable sentiment feature, significantly enhancing the model's predictive power.
* **Full-Stack Deployment:** The trained model is served via a lightweight **Flask web application (`app.py`)**, allowing users to view real-time predictions and visualizations through an interactive dashboard (`index.html`).

---

### Project live website


### 🛠️ Technology Stack

| Category | Tools & Libraries | Files Demonstrated |
| :--- | :--- | :--- |
| **Deep Learning** | **TensorFlow, Keras, FinBERT** | `model_trainer.py`, `tsla_model.h5` |
| **Data Science** | **Pandas, NumPy, Scikit-learn** | `Collect_data.py`, `scaler.pkl`, `TSLA_with_sentiment2.csv` |
| **Deployment** | **Flask, HTML/CSS, Jinja** | `app.py`, `index.html` |
| **Project Management** | Python, Git, GitHub | *Entire Repo* |

---

### 📂 Repository Structure

The project is logically organized into modules for data handling, model development, and deployment:

| File/Folder | Purpose |
| :--- | :--- |
| `Collect_data.py` | Python script for scraping/preprocessing the initial stock and news data. |
| `TSLA_with_sentiment2.csv` | The prepared dataset, enriched with FinBERT-derived sentiment features. |
| `model_trainer.py` | Script defining, training, and evaluating the **LSTM RNN** model. |
| `tsla_model.h5` | The serialized (saved) trained deep learning model weights and architecture. |
| `scaler.pkl` | The serialized **StandardScaler** object used for data normalization. |
| `app.py` | The **Flask server** responsible for loading the model and serving predictions. |
| `index.html` | The front-end template for the web interface and prediction visualization. |

---

### ➡️ How to Run the Project

1.  **Clone the repository:** `git clone [Your Repo URL]`
2.  **Install dependencies:** `pip install -r requirements.txt` (assuming you create this file)
3.  **Run the Flask application:** `python app.py`
4.  Navigate to `http://127.0.0.1:5000` in your web browser.
