from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    # Load data
    df_2021 = pd.read_csv('data/daily_receipts.csv', parse_dates=['date'])
    df_2021_monthly = df_2021.resample('M', on='date').sum()
    df_2022 = pd.read_csv('model/2022_predictions.csv', parse_dates=['date'])
    df_2022_monthly = df_2022.resample('M', on='date').sum()

    return render_template('index.html',
                           months_2021=df_2021_monthly.index.strftime('%Y-%m').tolist(),
                           counts_2021=df_2021_monthly['receipts'].tolist(),
                           months_2022=df_2022_monthly.index.strftime('%Y-%m').tolist(),
                           counts_2022=df_2022_monthly['receipts'].tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
