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
from io import StringIO

days_gone = (datetime.now()-datetime(2020,1,1)).days

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.129 Safari/537.36'}



# url = 'https://sampo.thl.fi/pivot/prod/fi/epirapo/covid19case/fact_epirapo_covid19case.csv?row=hcdmunicipality2020-445222&column=dateweek20200101-509030&column=508804L&filter=measure-444833&'

url = 'https://sampo.thl.fi/pivot/prod/fi/epirapo/covid19case/fact_epirapo_covid19case.csv?row=dateweek20200101-509030&row=509093L&row=hcdmunicipality2020-445222&column=measure-444833.445356.492118.&filter=measure-141082&&fo=1'


bytes_data = requests.get(url,headers=headers).content

s=str(bytes_data,'utf-8')

data = StringIO(s) 

data=pd.read_csv(data,sep=';')        

data=data[(data['Aika.1']!='Yhteensä') & (data.Mittari == 'Tapausten lukumäärä')].drop('Mittari',axis=1)
data=data.set_index('Aika.1')
data.index=pd.to_datetime(data.index)
data=data.dropna()
data.drop('Aika',axis=1,inplace=True)
data.index.name='pvm'
data.columns=['shp','infected']



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
                    html.P('Tällä sivulla testataan Lassoregression kykyä ennustaa uudet päivittäiset koronavirustartunnat sairaanhoitopiireittäin edeltävien tartuntatapausten perusteella sekä projisoidaan ennuste tulevaisuuteen. Data haetaan suoraan THL:n rajapinnasta (linkki alla), minkä jälkeen käyttäjän valitsemien edeltävien päivien tartuntatapausten määrän perusteella koneoppimisalgoritmi ennustaa tulevien päivien tartuntojen määrää. Näin käyttäjä voi itse valita opetusdatan, jolla saa parhaan ennusteen valitulle sairaanhoitopiirille testausindikaattorien muutosta seuraamalla. Tässä ratkaisussa hyödynnetään Scikit-learn kirjaston lineaarista Lassoregressiota (linkki tarkempaan dokumentaatioon alla). Tämän sivun tarkoitus on kokeilla kyseistä mallia tartuntojen ennustamiseen, ei niinkään tuottaa perustavanlaatuista ennustetta (vaikkakin esitettyjen laatumittareiden valossa tämä malli antaakin melko hyvän ennusteen tietyille sairaanhoitopiireille). Alla olevista valikoista voi valita sairaanhoitopiirin ja kuinka monelle päivälle ennusteen haluaa tehdä. Kuvaajan alla näkyy myös testin laatumittarit. Testi on tehty jakamalla data osiin (70% opetusdataa, 30% testausdataa). Itse mittareista voi lukea lisää Scikit-learn-kirjaston dokumentaatiosta (linkki alla).'),
        html.Br(),
                    html.Div(className = 'row',
                             children=[
                                        html.Div(className='six columns',children=[
                                            
                                                 html.H2('Valitse sairaanhoitopiiri.'),
                                                 dcc.Dropdown(id = 'sairaanhoitopiirit',
                                                              multi=False,
                                                              options = SHP,
                                                              value='Varsinais-Suomen SHP')
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
                    html.Div([html.H2('Valitse opetusdatan pituus'),
                            dcc.Slider(id ='alku', 
                                       min = 7,
                                       max = days_gone,
                                       step=1,
                                      value = 100,
                                      marks = {
                                      7: 'viikko',
                                      14: '2 viikkoa',
                                      30:'kuukausi',
                                      90: 'kolme kuukautta',
                                      180:'puoli vuotta',
                                      days_gone:'vuoden alusta'}),
                             html.Div(id = 'alku_indicator', style={'margin-top': 20})]),

                    html.Div(id='ennuste'),
                    html.Label(['Datan lähde: ', html.A('THL', href='https://thl.fi/fi/tilastot-ja-data/aineistot-ja-palvelut/avoin-data/varmistetut-koronatapaukset-suomessa-covid-19-')]),
                    html.Label(['Lassoregression dokumentaatio: ', html.A('Scikit-Learn', href='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#')]),
                    html.Label(['Regressiometriikoiden dokumentaatio: ', html.A('Scikit-Learn', href='https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics')]),
                    html.Label(['Katso toteutuneet koronatiedot ', html.A('täältä.', href='http://bit.ly/turkukorona'),
                    html.Label(['Tehnyt Tuomas Poukkula. ', html.A('Seuraa Twitterissä.', href='https://twitter.com/TuomasPoukkula')]),
                    html.Label(['Koodi Githubissa: ',html.A('Github',href='https://github.com/tuopouk/koronalasso')]),
                    ])
        
        
    ])

@app.callback(
    Output('alku_indicator', 'children'),
    [Input('alku', 'value')])
def update_year(value):
    
    return 'Valittu opetukseen: {} päivää ennen nykyhetkeä.'.format(
        str(value)
    )

@app.callback(
    Output('ennuste','children'),
    [Input('sairaanhoitopiirit','value'),
    Input('päivät','value'),
    Input('alku','value')]
)
def ennusta(shp,days,alku):
    
    global data

    days_gone =alku
    
  

    ratio=0.7
    lasso = Lasso(random_state=42)
    lasso_change = Lasso(random_state=42)
    scl = StandardScaler()


    df = pd.DataFrame(data[data.shp==shp].copy().resample('D').infected.sum().cumsum()).reset_index()# 
    
    
    dff = pd.DataFrame()
    dfff=df.copy().set_index('pvm')
    
    
    
    dff['pvm'] = dfff.index
    dff['next'] = dfff.index+pd.Timedelta(days=1)
    
    dff = dff.set_index('pvm')
    
    dff['infected'] = dfff.infected


    nexts = dfff.loc[[date for date in dff.next if date in dfff.index]]
    nexts = pd.concat([nexts, pd.DataFrame([{'pvm':nexts.index.max()+pd.Timedelta(days=1),'infected':np.nan}]).set_index('pvm')])

    dff['val'] = nexts.infected.values
    
    
    dff['muutos'] = dff.val-dff.infected
   

    df = dff.loc[datetime.today()-pd.Timedelta(days = days_gone):].reset_index()

    

    df_x = df.dropna().copy()
    

    x_train = df_x.iloc[:int(ratio*len(df_x))][['infected']]
    y_train = df_x.iloc[:int(ratio*len(df_x))]['muutos']
    X_train =scl.fit_transform(x_train)

    

    x_test = df_x.iloc[int(ratio*len(df_x)):][['infected']]
    y_test = df_x.iloc[int(ratio*len(df_x)):]['val']
    X_test = scl.transform(x_test)
    
    
    lasso_change.fit(X_train,y_train)

    y_hat_change=np.maximum(0,x_test['infected'] + lasso_change.predict(X_test))
    

    y_train = df_x.iloc[:int(ratio*len(df_x))]['val']
    

    lasso.fit(X_train,y_train)

    y_hat = np.maximum(0,lasso.predict(X_test))
    

    if mean_absolute_error(y_test, y_hat_change) < mean_absolute_error(y_test, y_hat):
        y_hat=y_hat_change
        lasso = lasso_change
        selected='Muutos'
    else:
        selected='Absoluuttinen'
    
    
    chain = 'Test (70/30 split): MAE: '+str(round(mean_absolute_error(y_test,y_hat),2))+', '
    chain = chain +'NMAE: '+str(round(mean_absolute_error(y_test,y_hat)/(y_test.std()+0.00000001),2))+', '
    chain = chain + 'RMSE: '+str(round(math.sqrt(mean_squared_error(y_test,y_hat)),2))+', '
    chain = chain + 'NRMSE '+str(round(math.sqrt(mean_squared_error(y_test,y_hat))/(y_test.std()+0.00000001),2))+', '
    chain = chain + 'R²: ' +str(round(r2_score(y_test,y_hat),2))+', ja '
    chain = chain + 'Explained variance score: '+str(round(explained_variance_score(y_test,y_hat),2))+'.'
    
    
    
    # Simuloi
    
    df_simulate = df.iloc[:int(ratio*len(df))]
    
    
    if selected == 'Muutos':
        

        for i in range(int(ratio*len(df)),len(df)):

            df_tail = df_simulate.copy().tail(1)

            
            df_tail.pvm =df_tail.next
            df_tail.next =df_tail.pvm+pd.Timedelta(days = 1)
            infected = df_tail.infected
            df_tail.infected = df_tail.val
            
            df_tail.muutos = lasso_change.predict(scl.transform(df_tail[['infected']]))
            df_tail.val = np.maximum(df_tail.infected,np.maximum(0, infected + df_tail.muutos))

            
            df_simulate = pd.concat([df_simulate, df_tail])
    else:
        
        for i in range(int(ratio*len(df)),len(df)):

            df_tail = df_simulate.copy().tail(1)
            
            df_tail.pvm =df_tail.next
            df_tail.next =df_tail.pvm+pd.Timedelta(days = 1)
            
            
            df_tail.infected = df_tail.val
            df_tail.val = np.maximum(df_tail.infected,np.maximum(0,  lasso.predict(scl.transform(df_tail[['infected']]))))

            
            df_simulate = pd.concat([df_simulate, df_tail])
    
    
        

    

    
    
    test_figure = go.Figure(data = [go.Scatter(x = df.pvm, y = df.infected, name = 'Toteutunut', mode = 'lines', marker = dict(color='green')),
                                               go.Scatter(x = df.iloc[int(ratio*len(df)):-1].next, y = np.ceil(y_hat), name = 'Ennuste', mode = 'lines+markers', marker = dict(color =  'red'))],
                           layout = go.Layout(title = dict(text = 'Lyhyen tähtäimen ennusteen ja toteuman välinen ero sairaanhoitopiirille: <br>'+shp,
                                                                                   y = 0.9,
                                                                                   x = 0.5,
                                                           font=dict(size=18),
                                                                                   xanchor = 'center',
                                                                                   yanchor = 'top'
                                                                               ),
                                                                          yaxis = dict(title = 'Tartunnat', tickformat =' '),
                                                                          xaxis = dict(title = 'Päivät'),
                                                                          hovermode="x unified",
                                                                          autosize = True,
                                                                             font=dict(family='Arial',
                                                                                         size=16,
                                                                                         color='black'
                                                                )
                                             )
                           )

    y_test = df.iloc[int(ratio*len(df))-1:-1].val
    y_hat = df_simulate.iloc[int(ratio*len(df))-1:-1].val
    sim_chain = 'Test (70/30 split): MAE: '+str(round(mean_absolute_error(y_test,y_hat),2))+', '
    sim_chain = sim_chain +'NMAE: '+str(round(mean_absolute_error(y_test,y_hat)/(y_test.std()+0.00000001),2))+', '
    sim_chain = sim_chain + 'RMSE: '+str(round(math.sqrt(mean_squared_error(y_test,y_hat)),2))+', '
    sim_chain = sim_chain + 'NRMSE '+str(round(math.sqrt(mean_squared_error(y_test,y_hat))/(y_test.std()+0.00000001),2))+', '
    sim_chain = sim_chain + 'R²: ' +str(round(r2_score(y_test,y_hat),2))+', ja '
    sim_chain = sim_chain + 'Explained variance score: '+str(round(explained_variance_score(y_test,y_hat),2))+'.'
    simulate_figure = go.Figure(data = [go.Scatter(x = df.pvm, y = df.infected, name = 'Toteutunut', mode = 'lines', marker = dict(color='green')),
                                               go.Scatter(x = df_simulate.iloc[int(ratio*len(df))-1:-1].next, y = np.ceil(df_simulate.iloc[int(ratio*len(df))-1:-1].val), name = 'Ennuste', mode = 'lines+markers', marker = dict(color =  'red'))],
                           layout = go.Layout(title = dict(text = 'Pitkän tähtäimen ennusteen ja toteuman välinen ero sairaanhoitopiirille: <br>'+shp,
                                                                                   y = 0.9,
                                                                                   x = 0.5,
                                                           font=dict(size=18),
                                                                                   xanchor = 'center',
                                                                                   yanchor = 'top'
                                                                               ),
                                                                          yaxis = dict(title = 'Tartunnat', tickformat =' '),
                                                                          xaxis = dict(title = 'Päivät'),
                                                                          hovermode="x unified",
                                                                          autosize = True,
                                                                             font=dict(family='Arial',
                                                                                         size=16,
                                                                                         color='black'
                                                                )
                                             )
                           )
    
    
    datapoints = days
    
    if df.val.isna().sum() > 0:
        max_date = df.pvm.max()
    else:
        max_date = df.next.max()
    
    
    x_train = df_x[['infected']]
    
    X_train =scl.fit_transform(x_train)
    
    if selected == 'Muutos':
        
        
        y_train = df_x[['muutos']]
        
        
        lasso_change.fit(X_train, y_train)

        
        for i in range(datapoints):
            
            df_tail = df.tail(1).copy()

            
            if i == 0:
                
                df = df.iloc[:-1,:]

            
            else:
                df_tail.pvm = df_tail.next
                df_tail.next = df_tail.pvm + pd.Timedelta(days = 1)
                df_tail.infected = df_tail.val
                
                
            x_predict = df_tail[['infected']]
            X_predict = scl.transform(x_predict)
            
            
            df_tail.muutos = lasso.predict(X_predict)
            df_tail.val = np.maximum(df_tail.infected,np.maximum(0, df_tail.infected + df_tail.muutos))

            df_tail.index+=1
            
           
            
            df = pd.concat([df,df_tail])
            
    
    else:
        
        y_train = df_x[['val']]
        
        lasso.fit(X_train, y_train)
        

        for i in range(datapoints):
            
            df_tail = df.tail(1).copy()
            
            
            if i==0:
                df = df.iloc[:-1,:]
            
            else:
                
                df_tail.pvm=df_tail.next
                df_tail.next = df_tail.next + pd.Timedelta(days = 1)
                df_tail.infected = df_tail.val
                
                df_tail.index+=1
            
            x_predict = df_tail[['infected']]
            

            
            X_predict = scl.transform(x_predict)
        
            

            df_tail.val = np.maximum(df_tail.infected,np.maximum(0,  lasso.predict(X_predict)))
           
            df_tail.index+=1
            

      
            
            df = pd.concat([df,df_tail])
            

        
    
    
    
    
    
    df =df.set_index('pvm')

    
    df.infected = np.ceil(df.infected)
    
    
    df_days = df['infected'].diff()
    
              
               
    
    return html.Div(children=[
                            html.Br(),
                            #html.P('Laatuparametrit',style=dict(textAlign='center',fontSize=30, fontFamily='Arial')),
                            html.P(chain,style=dict(textAlign='center',fontSize=25, fontFamily='Arial')),
                            #html.Br(),
                            html.P('Ennuste',style=dict(textAlign='center',fontSize=30, fontFamily='Arial')),
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
                                                          text = 'Koronavirustartuntojen seuraavan '+str(days)+' päivän kumulatiivinen ennuste alueelle: <br>'+shp,
                                                                                   y = 0.9,
                                                                                   x = 0.5,
                                                                                   xanchor = 'center',
                                                                                   yanchor = 'top'
                                                                                  ),
                                                                           hovermode="x unified",
                                                                          yaxis = dict(title = 'Tartunnat', tickformat =' '),
                                                                          xaxis = dict(title = 'Päivät'),
                                                                          autosize = True,
                                                                          font=dict(family='Arial',
                                                                 size=16,
                                                                 color='black'
                                                                ))
                                                         )
                                      ),
                                                         
                                
                                html.Br(),
                                dcc.Graph(config={'modeBarButtonsToRemove':['sendDataToCloud']},
                                         figure = go.Figure(data= [
                                                                 go.Bar(x = df_days.loc[:max_date].index,
                                                                       y = np.maximum(0,df_days.loc[:max_date].values),
                                                                       name = 'Toteutunut',
                                                                       text = np.maximum(0,df_days.loc[:max_date].values),
                                                                       textposition='outside'),
                                                                 go.Bar(x = df_days.loc[max_date+pd.Timedelta(days=1):].index,
                                                                       y = np.maximum(0,df_days.loc[max_date+pd.Timedelta(days=1):].values),
                                                                       name = 'Ennuste',
                                                                       text = np.maximum(0,df_days.loc[max_date+pd.Timedelta(days=1):].values),
                                                                       textposition='outside'
                                                                       )],
                                                           layout = go.Layout(title = dict(
                                                                    text = 'Koronavirustartuntojen seuraavan '+str(days)+' päivän päivittäiset ennusteet alueelle: <br>'+shp,
                                                                                   y = 0.9,
                                                                                   x = 0.5,
                                                                                   xanchor = 'center',
                                                                                   yanchor = 'top'
                                                                               ),
                                                                              hovermode="x unified",
                                                                          yaxis = dict(title = 'Tartunnat', tickformat =' '),
                                                                          xaxis = dict(title = 'Päivät'),
                                                                          autosize = True,
                                                                             font=dict(family='Arial',
                                                                 size=16,
                                                                 color='black'
                                                                )))),
        html.Br(),
        html.P("Testi ja simulaatio",style=dict(textAlign='center',fontSize=30, fontFamily='Arial')),
        html.P('Testillä ennustetaan seuraavan päivän tartuntoja edellisen päivän toteuman perusteella. Näin voidaan tarkastella "huomisen" ennusteen laatua.',style=dict(textAlign='center',fontSize=15, fontFamily='Arial')),
        html.P('Simulaatiolla ennustetaan seuraavan päivän tartuntoja edellisen päivän ennusteen perusteella. Näin voidaan tarkastella pidemmän tulevaisuuden ennusteen laatua.',style=dict(textAlign='center',fontSize=15, fontFamily='Arial')),

        html.Div(className='row', children=[  
            html.Div(className='six columns',children=[
                html.Br(),
                dcc.Graph(config={'modeBarButtonsToRemove':['sendDataToCloud']},
                          figure=test_figure),
                html.P(chain,style=dict(textAlign='center',fontSize=12, fontFamily='Arial')),
            ]),
            html.Div(className='six columns',
                     children=[
                html.Br(),
                dcc.Graph(config={'modeBarButtonsToRemove':['sendDataToCloud']},
                          figure=simulate_figure),
                         html.P(sim_chain,style=dict(textAlign='center',fontSize=12, fontFamily='Arial')),
            ])
        
    ])

    ]
                   )


app.layout= serve_layout
#Aja sovellus.
if __name__ == '__main__':
    app.run_server(debug=False)
