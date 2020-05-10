# -*- coding: utf-8 -*-

import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,  explained_variance_score, mean_squared_log_error,median_absolute_error,r2_score
import warnings
import plotly.graph_objs as go
from sklearn.linear_model import Lasso
import requests
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from flask import Flask
from datetime import datetime
import os

days_gone = 45

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.129 Safari/537.36'}



url = 'https://sampo.thl.fi/pivot/prod/fi/epirapo/covid19case/fact_epirapo_covid19case.json?row=dateweek2020010120201231-443702L&column=hcdmunicipality2020-445222L'

pvm =pd.DataFrame(list(requests.get(url,headers=headers).json()['dataset']['dimension']['dateweek2020010120201231']['category']['label'].values()),columns=['pvm'])
pvm['key']=0
shp = pd.DataFrame(list(requests.get(url,headers=headers).json()['dataset']['dimension']['hcdmunicipality2020']['category']['label'].values()),columns=['shp'])
shp['key']=0
data = pd.merge(left=pvm,right=shp).drop('key',axis=1)
data.index=data.index.astype(int)
vals = pd.DataFrame([requests.get(url,headers=headers).json()['dataset']['value']]).T.rename(columns={0:'infected'})
vals.index=vals.index.astype(int)
data=pd.merge(left=data,right=vals,left_on=data.index, right_on=vals.index,how='right').drop('key_0',axis=1)
data.pvm=pd.to_datetime(data.pvm)
data=data.set_index('pvm')
data.infected=data.infected.astype(float)


SHP = [{'label':s, 'value': s} for s in sorted(list(pd.unique(data.shp)))]

server = Flask(__name__)
server.secret_key = os.environ.get('secret_key','secret')
app = dash.Dash(name = __name__, server = server)



app.title = 'KoronaLasso'

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            <script src="https://cdn.plot.ly/plotly-locale-fi-latest.js"></script>
<script>Plotly.setPlotConfig({locale: 'fi'});</script>
          {%renderer%}
          </footer>
          
            </body>
</html>
'''

#app.scripts.append_script({"external_url": "https://raw.githubusercontent.com/plotly/plotly.js/master/dist/plotly-locale-fi.js"})

app.config['suppress_callback_exceptions']=True


 


ratio=0.7
lasso = Lasso(random_state=42)
scl = StandardScaler()



def serve_layout():
    


    SHP = [{'label':s, 'value': s} for s in sorted(list(pd.unique(data.shp)))]
    

    
    return html.Div(children = [

                    html.H1('KoronaLasso',style=dict(textAlign='center')),
                    html.Br(),
                    html.P('Tällä sivulla testataan Lassoregression kykyä ennustaa uudet päivittäiset koronavirustartunnat sairaanhoitopiireittäin edeltävien tartuntatapausten perusteella sekä projisoidaan ennuste tulevaisuuteen. Data haetaan suoraan THL:n rajapinnasta (linkki alla), minkä jälkeen '+str(days_gone)+' edeltävien tartuntatapausten määrän perusteella koneoppimisalgoritmi ennustaa tulevien päivien tartuntojen määrää. Tässä ratkaisussa hyödynnetään Scikit-learn kirjaston lineaarista Lassoregressiota (linkki tarkempaan dokumentaatioon alla). Tämän sivun tarkoitus on kokeilla kyseistä mallia tartuntojen ennustamiseen, ei niinkään tuottaa perustavanlaatuista ennustetta (vaikkakin esitettyjen laatumittareiden valossa tämä malli antaakin melko hyvän ennusteen tietyille sairaanhoitopiireille). Alla olevista valikoista voi valita sairaanhoitopiirin ja kuinka monelle päivälle ennusteen haluaa tehdä. Kuvaajan alla näkyy myös testin laatumittarit. Testi on tehty jakamalla data osiin (70% opetusdataa, 30% testausdataa). Itse mittareista voi lukea lisää Scikit-learn-kirjaston dokumentaatiosta (linkki alla).'),
        html.Br(),
                    html.Div(className = 'row',
                             children=[
                                        html.Div(className='six columns',children=[
                                            
                                                 html.H2('Valitse sairaanhoitopiiri.'),
                                                 dcc.Dropdown(id = 'sairaanhoitopiirit',
                                                              multi=False,
                                                              options = SHP,
                                                              value='Kaikki Alueet')
                                                ]),
                                        html.Div(className='six columns',children=[
                                                html.H2('Valitse ennusteen pituus (päivää).'),
                                                dcc.Slider(id='päivät',
                                                           min=1,
                                                           max=100,
                                                           step=1,
                                                           value=10,
                                                           marks = {
                                                           1: '1 päivä',
                                                           10: '10 päivää',
                                                           30: '30 päivää',
                                                           50: '50 päivää',
                                                           90: '90 päivää',
                                                           100: '100 päivää'}
                                                          )
                                                ])
                             ]),

                    html.Div(id='ennuste'),
                    html.Label(['Datan lähde: ', html.A('THL', href='https://thl.fi/fi/tilastot-ja-data/aineistot-ja-palvelut/avoin-data/varmistetut-koronatapaukset-suomessa-covid-19-')]),
                    html.Label(['Lassoregression dokumentaatio: ', html.A('Scikit-Learn', href='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#')]),
                    html.Label(['Regressiometriikoiden dokumentaatio: ', html.A('Scikit-Learn', href='https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics')]),
                    html.Label(['by Tuomas Poukkula ', html.A('Twitter', href='https://twitter.com/TuomasPoukkula')]),
                    html.Label(['Katso toteutuneet koronatiedot ', html.A('täältä.', href='http://bit.ly/turkukorona')])
        
        
    ])

@app.callback(
    Output('ennuste','children'),
    [Input('sairaanhoitopiirit','value'),
    Input('päivät','value')]
)
def ennusta(shp,days):


    
    url = 'https://sampo.thl.fi/pivot/prod/fi/epirapo/covid19case/fact_epirapo_covid19case.json?row=dateweek2020010120201231-443702L&column=hcdmunicipality2020-445222L'
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.129 Safari/537.36'}
    pvm =pd.DataFrame(list(requests.get(url,headers=headers).json()['dataset']['dimension']['dateweek2020010120201231']['category']['label'].values()),columns=['pvm'])
    pvm['key']=0
    shp_ = pd.DataFrame(list(requests.get(url,headers=headers).json()['dataset']['dimension']['hcdmunicipality2020']['category']['label'].values()),columns=['shp'])
    shp_['key']=0
    data = pd.merge(left=pvm,right=shp_).drop('key',axis=1)
    data.index=data.index.astype(int)
    vals = pd.DataFrame([requests.get(url,headers=headers).json()['dataset']['value']]).T.rename(columns={0:'infected'})
    vals.index=vals.index.astype(int)
    data=pd.merge(left=data,right=vals,left_on=data.index, right_on=vals.index,how='right').drop('key_0',axis=1)
    data.pvm=pd.to_datetime(data.pvm)
    data=data.set_index('pvm')
    data.infected=data.infected.astype(float)

    ratio=0.7
    lasso = Lasso(random_state=42)
    scl = StandardScaler()


    df = pd.DataFrame(data[data.shp==shp].resample('D').infected.sum().cumsum()).reset_index()#              
    df_x = df[df.index % 2 == 0]
    df_y = df[df.index % 2 != 0]
    df = pd.concat([df_x.reset_index(),df_y.reset_index()],axis=1)
    df.columns=['index','pvm','infected','i','next','val']
    df.drop('i',axis=1,inplace=True)
    df.next = df.next.fillna(df.pvm.max()+pd.Timedelta(days=1))
    df['muutos'] = df.val-df.infected
    df = df.set_index('pvm').sort_index().loc[datetime.today()-pd.Timedelta(days = days_gone):].reset_index()
    df.next = df.next.fillna(df.pvm + pd.Timedelta(days=1))
    #print(df.tail())

    df_x = df.dropna().copy()


    x_train = df_x.iloc[:int(ratio*len(df_x))][['infected']]
    y_train = df_x.iloc[:int(ratio*len(df_x))]['muutos']
    X_train =scl.fit_transform(x_train)

    

    x_test = df_x.iloc[int(ratio*len(df_x)):][['infected']]
    y_test = df_x.iloc[int(ratio*len(df_x)):]['val']
    X_test = scl.transform(x_test)
    
    
    lasso.fit(X_train,y_train)

    y_hat_change=np.maximum(0,x_test['infected'] + lasso.predict(X_test))
    

    y_train = df_x.iloc[:int(ratio*len(df_x))]['val']
    

    lasso.fit(X_train,y_train)

    y_hat = np.maximum(0,lasso.predict(X_test))
    

    if mean_absolute_error(y_test, y_hat_change) < mean_absolute_error(y_test, y_hat):
        y_hat=y_hat_change
        selected='Muutos'
    else:
        selected='Absoluuttinen'
    
    
    chain = 'Test (70/30 split): MAE: '+str(round(mean_absolute_error(y_test,y_hat),2))+', '
    chain = chain +'NMAE: '+str(round(mean_absolute_error(y_test,y_hat)/(y_test.std()+0.00000001),2))+', '
    chain = chain + 'RMSE: '+str(round(math.sqrt(mean_squared_error(y_test,y_hat)),2))+', '
    chain = chain + 'NRMSE '+str(round(math.sqrt(mean_squared_error(y_test,y_hat))/(y_test.std()+0.00000001),2))+', '
    chain = chain + 'R²: ' +str(round(r2_score(y_test,y_hat),2))+', ja '
    chain = chain + 'Explained variance score: '+str(round(explained_variance_score(y_test,y_hat),2))+'.'
    
    

    
    datapoints = days
    
    if df.val.isna().sum() > 0:
        max_date = df.pvm.max()
    else:
        max_date = df.next.max()
    #print(max_date)
    
    x_train = df_x[['infected']]
    X_train =scl.fit_transform(x_train)
    
    if selected == 'Muutos':
        
        
        y_train = df_x[['muutos']]
        
        lasso.fit(X_train, y_train)

        
        for i in range(datapoints):
            
            df_tail = df.tail(1).copy()
            
            if df_tail.val.notna().values[0]:
            
                df_tail.pvm=df_tail.next
                df_tail.next = df_tail.next + pd.Timedelta(days = 1)
                df_tail.infected = df_tail.val
                df_tail['index']+=1
                df_tail.index+=1
            
                
                

            x_predict = df_tail[['infected']]
            X_predict = scl.transform(x_predict)
            
            df_tail.muutos = lasso.predict(X_predict)
            df_tail.val = np.maximum(0, df_tail.infected + df_tail.muutos)
            df = pd.concat([df,df_tail])
    
    else:
        
        y_train = df_x[['val']]
        
        lasso.fit(X_train, y_train)
        

        for i in range(datapoints):
            
            df_tail = df.tail(1).copy()
            
            if df_tail.val.notna().values[0]:
            
                df_tail.pvm=df_tail.next
                df_tail.next = df_tail.next + pd.Timedelta(days = 1)
                df_tail.infected = df_tail.val
                df_tail['index']+=1
                df_tail.index+=1
            
            x_predict = df_tail[['infected']]
            X_predict = scl.transform(x_predict)
        
            df_tail.val = lasso.predict(X_predict)
            df_tail.val = np.maximum(0,  df_tail.val)
            df = pd.concat([df,df_tail])

        
    #df = df.set_index('pvm')
    df = pd.concat([df[['pvm','infected']],df[['next','val']].rename(columns={'next':'pvm','val':'infected'})],axis=0).drop_duplicates().set_index('pvm').sort_index()
    
    df.infected = np.ceil(df.infected)
    
    df_days = df['infected'].diff().dropna()
    
    #df_days.iloc[0] = df.iloc[0]
    
    
    
    return html.Div(children=[
                             dcc.Graph(config={'modeBarButtonsToRemove':['sendDataToCloud']},
                                       figure = go.Figure(data=[
                                                                go.Scatter(x = df.loc[:max_date].index, 
                                                                           y = df.loc[:max_date].infected,
                                                                           name='Toteutunut'),
                                                                go.Scatter(x = df.loc[max_date+pd.Timedelta(days=1):].index,
                                                                           y = df.loc[max_date+pd.Timedelta(days=1):].infected,
                                                                           name='Ennuste')
                                                               ],
                                                          layout=go.Layout(title = dict(
                                                                                   text = str(days)+' päivän ennuste alueelle: '+shp,
                                                                                   y = 0.9,
                                                                                   x = 0.5,
                                                                                   xanchor = 'center',
                                                                                   yanchor = 'top'
                                                                                  ),
                                                                          yaxis = dict(title = 'Tartunnat', tickformat =' '),
                                                                          xaxis = dict(title = 'Päivät'),
                                                                          autosize = True)
                                                         )
                                      ),
                                                         
                                html.P(chain),
                                html.Br(),
                                dcc.Graph(config={'modeBarButtonsToRemove':['sendDataToCloud']},
                                         figure = go.Figure(data= [
                                                                 go.Bar(x = df_days.loc[:max_date].index,
                                                                       y = df_days.loc[:max_date].values,
                                                                       name = 'Toteutunut'),
                                                                 go.Bar(x = df_days.loc[max_date+pd.Timedelta(days=1):].index,
                                                                       y = df_days.loc[max_date+pd.Timedelta(days=1):].values,
                                                                       name = 'Ennuste')],
                                                           layout = go.Layout(title = dict(
                                                                    text = str(days)+' päivän päivittäiset ennusteet alueelle: '+shp,
                                                                                   y = 0.9,
                                                                                   x = 0.5,
                                                                                   xanchor = 'center',
                                                                                   yanchor = 'top'
                                                                               ),
                                                                          yaxis = dict(title = 'Tartunnat', tickformat =' '),
                                                                          xaxis = dict(title = 'Päivät'),
                                                                          autosize = True)
                                                           )
                                         )
    ])

app.layout= serve_layout
#Aja sovellus.
if __name__ == '__main__':
    app.run_server(debug=False)