# 📈📉 Sales Prediction using Time Series Analysis

This project focuses on performing time series analysis on sales data to identify trends, patterns, and forecast future sales performance. The goal is to support data-driven business decisions around planning, budgeting, and strategic growth. The analysis and modeling are conducted using Python in a Jupyter Notebook, with final models and forecasts saved in the `models/` folder for future use and evaluation.

## 📁 Project Structure

<pre>
.
├── data/ Data set links
├── models/ # Saved trained models (pickle file)
├── notebooks/ # Jupyter Notebook used for analysis
│ └──  sales_prediction.ipynb
├── README.md # This file
└── .gitignore # Files to ignore
</pre>

##  📊 Dataset

Refer to the `data/` folder for details

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/Mohammed-Hanan-Othman/time-series-forecasting.git
cd time-series-forecasting
```

2. **Installation and Setup:**
This assumes an anaconda environment.

3. **Open notebook**
``` bash
jupyter notebook notebooks/sales_prediction.ipynb
```

4. **Run:**
    - Run all cells to reproduce the results.
    - Alternatively, create a new `.py` file and load the saved models from the `.models/` folder to make predictions


## 🧠 Model Overview

Different models were  used here:

1. **Base Model:**
    - Model type: ARIMA (order=(5,1,0)).
    - Notebook: [Base Model](./notebooks/sales_prediction.ipynb)
    - Evaulation Metrics:
        - Root Mean Square Error
    - Model Output: Model trained and saved as a `.pkl` file in `.models/` folder.
    - 📈 Results:
        - Walk forward validation RMSE: 74464.67

## 📌 Future Improvements

## Contributing
Contributions are welcome. To contribute:

- Fork the repository
- Create a new branch (git checkout -b feature-xyz)
- Commit your changes (git commit -am 'Add new feature')
- Push to the branch (git push origin feature-xyz)
- Open a Pull Request

Contributors
- GitHub: @Mohammed-Hanan-Othman – Initial work

Feel free to open issues or suggestions via GitHub Issues.

## 📄 License
This project is for educational purposes only. Feel free to use and share.

## 👨‍💻 Author
- GitHub: @Mohammed-Hanan-Othman