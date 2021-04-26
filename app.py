
from flask import Flask,render_template,redirect,request
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from fbprophet import Prophet

import warnings
warnings.filterwarnings('ignore')
from random import randint
import plotly.graph_objs as go
import plotly.offline as py
from flask_socketio import SocketIO
import datetime
from datetime import timedelta, date
from fbprophet.plot import plot_plotly


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)
    

hist_df = pd.read_csv('hkd_data_2010_inr.csv')  
start_date = date(2021, 4, 23)
end_date = datetime.date.today()
df = pd.DataFrame()
for single_date in daterange(start_date, end_date):
    dfs = pd.read_html(f'https://www.xe.com/currencytables/?from=HKD&date={single_date.strftime("%Y-%m-%d")}')[0]
    dfs['Date'] = single_date.strftime("%Y-%m-%d")
    df = df.append(dfs)  
df.to_csv('hkd_data.csv')
inr_df = df[df['Currency'] == 'INR']
inr_df.pop('Rate')
inr_df.pop('Change')
inr_df.head(5)
inr_df = pd.concat([hist_df, inr_df], ignore_index=True)

length=len(inr_df)
data_day1=inr_df[length-1:]
data_day2=inr_df[length-2:length-1]
data_day7=inr_df[length-7:length-6]
data_day15=inr_df[length-15:length-14]
data_day365=inr_df[length-365:length-364]

change_1=float(data_day2['Units per HKD'])-float(data_day1['Units per HKD'])
change_7=float(data_day7['Units per HKD'])-float(data_day1['Units per HKD'])
change_15=float(data_day15['Units per HKD'])-float(data_day1['Units per HKD'])
change_365=float(data_day365['Units per HKD'])-float(data_day1['Units per HKD'])

price_day1=float(data_day1['Units per HKD'])
print(price_day1,change_1,change_7,change_15,change_365)




import os
app = Flask("__name__")
app.config["IMAGE_UPLOADS"] = "static/img/"
socketio = SocketIO(app)
@app.route('/')
def hello():
    
    
    return render_template("step1.html",price_day1=price_day1,change_1=change_1,change_7=change_7,change_15=change_15,change_365=change_365)







@app.route('/submit',methods=['POST'])
def submit_data():
    
        
    print("entered")
    s1=request.form.getlist('options')[0]
    s2=int(request.form['parameter'])
    print(s1,s2)

    df= inr_df.drop(['Currency', 'Name', 'HKD per unit'], axis=1)

    df = df.rename(columns={'Units per HKD': 'y', 'Date': 'ds'})
    #df['ds'] =  pd.to_datetime(df['ds'], format='%d/%m/%Y')
    df.head(5)


    # to save a copy of the original data..you'll see why shortly. 
    df['y_orig'] = df['y'] 
    # log-transform of y
    df['y'] = np.log(df['y'])
    #instantiate Prophet
    model = Prophet() 
    model.fit(df)
    future_data = model.make_future_dataframe(periods=s2, freq = s1)  #dropdown   
    future_data.tail()
    forecast_data = model.predict(future_data)
    forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)



    # make sure we save the original forecast data
    forecast_data_orig = forecast_data 
    forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
    forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
    forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])
    df['y_log']=df['y'] 
    df['y']=df['y_orig']
    final_df = pd.DataFrame(forecast_data_orig)


    '''
    actual_chart = go.Scatter(y=df["y_orig"], name= 'Actual')
    predict_chart = go.Scatter(y=final_df["yhat"], name= 'Predicted')
    predict_chart_upper = go.Scatter(y=final_df["yhat_upper"], name= 'Predicted Upper')
    predict_chart_lower = go.Scatter(y=final_df["yhat_lower"], name= 'Predicted Lower')
    '''  
    
    fig,ax=plt.subplots(nrows=1, ncols=1)
    ax.plot(df["y_orig"], label= 'Actual')
    ax.plot(final_df["yhat"], label= 'Predicted')
    ax.plot(final_df["yhat_lower"], label= 'Predicted Lower')
    ax.plot(final_df["yhat_upper"], label= 'Predicted Upper')
    ax.legend()


    plt.xticks(rotation=90)

    n=randint(0,1000000000000)
    n=str(n)
    fig.savefig(os.path.join(app.config["IMAGE_UPLOADS"],n+'time_series.png'))  
    full_filename= os.path.join(app.config["IMAGE_UPLOADS"],n+'time_series.png')                  
    
    
    return render_template("step1.html",user_image = full_filename,price_day1=price_day1,change_1=change_1,change_7=change_7,change_15=change_15,change_365=change_365)


   

    
if __name__ =="__main__":

    socketio.run(app)
    
