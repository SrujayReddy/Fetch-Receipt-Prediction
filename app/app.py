from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)
@app.route('/')
def dashboard():
    # Absolute paths for Docker
    historical = pd.read_csv('/app/data/daily_receipts.csv', parse_dates=['Date'])
    predictions = pd.read_csv('/app/model/2022_predictions.csv', parse_dates=['Date'])

    hist_monthly = historical.resample('M', on='Date').sum()
    pred_monthly = predictions.resample('M', on='Date').sum()

    return render_template('index.html',
                         hist_labels=hist_monthly.index.strftime('%Y-%m').tolist(),
                         hist_data=hist_monthly['Receipt_Count'].tolist(),
                         pred_labels=pred_monthly.index.strftime('%Y-%m').tolist(),
                         pred_data=pred_monthly['Predicted_Receipts'].tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Port changed to 5001
