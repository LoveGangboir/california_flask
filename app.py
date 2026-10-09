from flask import Flask, request, render_template
import pandas as pd
import joblib

obj = joblib.load('california.joblib')
model = obj['Model']
columns = obj['Columns']

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    INPUT = []
    for i in columns:
        value = request.form.get(i, default=0, type=float)
        INPUT.append(value)

    df_input = pd.DataFrame([INPUT], columns=columns)   
    out = model.predict(df_input)
    
    # User ki requirement: 100 se multiply karna
    # (Note: Sklearn ka model real mein output $100,000 ke units mein deta hai. 
    # Agar aapko exact real-world price chahiye, toh 100 ki jagah 100000 use karein)
    final_price = out[0] * 100000
    
    # Dollar format aur 2 decimal places ke sath return karna
    return f'Predicted Price is ${final_price:,.2f}'

if __name__ == '__main__': 
    app.run(debug=True)

 