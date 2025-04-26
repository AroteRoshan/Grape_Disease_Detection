from flask import Flask, render_template, request, send_from_directory
import cv2
import pickle
import joblib
import numpy as np
from flask import Flask, request, jsonify, abort
import requests
from bs4 import BeautifulSoup
from datetime import datetime

from keras.models import load_model

#load model

model_grape= load_model("grape_model_final.h5")


COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/leaf_detection')
def leaf_detection():
    return render_template('leaf_detection.html')


@app.route('/inputgrape')
def inputgrape():
    return render_template('prediction_Grape.html')

@app.route('/learn_more')
def learn_more():
    return render_template('/learn_more.html')
@app.route('/learn1')
def learn1():
    return render_template('/learn1.html')



@app.route('/data' , methods = ['POST','GET'])
def submit():
    if request.method == 'POST':
        name = request.form['name']
        phone = int(request.form['phone'])
        email = request.form['email']
        subject =request.form['subject']
        message =request.form['message']

        print("Name Of User:",name)
        print("Phone no:",phone)
        print("Email:",email)
        print("subject:",subject)
        print("message:",message)

        return render_template('index.html')
    
    else :
        return render_template('index.html')




@app.route('/predictiongrape',methods = ['POST'])
def predictiongrape():
    global COUNT
    img = request.files['image']

    img.save('static/img/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/img/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (256, 256))
    #img_arr = img_arr / 255.0
    #img_arr = img_arr.reshape(1, 256, 256, 3)
    img_arr=np.array([img_arr])
    predictions = model_grape.predict(img_arr)
    prediction=np.argmax(predictions)
    print(prediction)
    #
    # x = round(prediction[0])
    # # y = round(prediction[0, 1], 2)
    # preds = np.array([x])
    COUNT += 1
    if prediction == 0:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Healthy", 'green'])


    elif prediction== 1:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Black Rot", 'red'])
    elif prediction== 2:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Grape___Esca_(Black_Measles)", 'red'])
    else:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 'red'])

@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static/img', "{}.jpg".format(COUNT-1))


COMMODITY = 22
STATE     = "MH"
DISTRICT  = 13
MARKET    = 168

BASE_URL = (
    "https://agmarknet.gov.in/SearchCmmMkt.aspx"
    "?Tx_Commodity={c}&Tx_State={s}&Tx_District={d}&Tx_Market={m}"
    "&DateFrom={df}&DateTo={dt}&Fr_Date={df}&To_Date={dt}"
    "&Tx_Trend=0&Tx_CommodityHead=Grapes&Tx_StateHead=Maharashtra"
    "&Tx_DistrictHead=Nashik&Tx_MarketHead=Nasik"
)

def _validate_date(d):
    try:
        # expecting e.g. 25-Apr-2025
        return datetime.strptime(d, "%d-%b-%Y")
    except ValueError:
        return None

def fetch_market_data(datefrom: str, dateto: str) -> dict:
    if not (datefrom and dateto):
        raise ValueError("Both `datefrom` and `dateto` are required")
    if not (_validate_date(datefrom) and _validate_date(dateto)):
        raise ValueError("Dates must be in DD-MMM-YYYY format (e.g. 25-Apr-2025)")

    url = BASE_URL.format(
        c=COMMODITY, s=STATE, d=DISTRICT, m=MARKET,
        df=datefrom, dt=dateto
    )

    resp = requests.get(url)
    if resp.status_code != 200:
        raise ConnectionError(f"Agmarknet returned {resp.status_code}")

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id="cphBody_GridPriceData")
    if not table:
        raise RuntimeError("Data table not found on Agmarknet page")

    rows = table.find_all("tr")[1:]  # skip header row
    records = []
    for tr in rows:
        cols = tr.find_all("td")
        if len(cols) < 10:
            continue
        records.append({
            "sl_no":         cols[0].get_text(strip=True),
            "district":      cols[1].get_text(strip=True),
            "market":        cols[2].get_text(strip=True),
            "commodity":     cols[3].get_text(strip=True),
            "variety":       cols[4].get_text(strip=True),
            "grade":         cols[5].get_text(strip=True),
            "min_price":     cols[6].get_text(strip=True),
            "max_price":     cols[7].get_text(strip=True),
            "modal_price":   cols[8].get_text(strip=True),
            "price_date":    cols[9].get_text(strip=True),
        })

    return {
        "query":   {"datefrom": datefrom, "dateto": dateto},
        "records": records
    }


@app.route("/market-data")
def market_data_view():
    datefrom = request.args.get("datefrom")
    dateto   = request.args.get("dateto")

    data = {}
    # try:
    data = fetch_market_data(datefrom, dateto)
    # except Exception:
    #     data = {"query": {"datefrom": datefrom, "dateto": dateto}, "records": []}

    return render_template("market_data.html",
                           query=data.get("query", {}),
                           records=data.get("records", []))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)


