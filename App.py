import streamlit as st
import json
from fbprophet import Prophet
from prophet.serialize import model_from_json
import pandas as pd
from prophet.plot import plot_plotly

def load_model():
    with open('modelo_o2_prophet.json', 'r') as file_in:
        modelo = model_from_json(json.load(file_in))  # Load model
        return modelo
    
modelo = load_model()

st.title('Previsão de Séries Temporais - Modelo Prophet')   

st.caption('Aplicação de Previsão de Séries Temporais utilizando o modelo Prophet')

st.subheader('Insira o número de dias para a previsão:')

dias = st.number_input(' ',min_value=1, value=1, step=1)

if 'previsao_feita' not in st.session_state:
    st.session_state['previsao_feita'] = False
    st.session_state['dados_previsao'] = None

if st.button('Fazer Previsão'):
    st.session_state['previsao_feita'] = True
    futuro = modelo.make_future_dataframe(periods=dias, freq='D')
    previsao = modelo.predict(futuro)
    st.session_state['dados_previsao'] = previsao

if st.session_state.previsao_feita:
    fig = plot_plotly(modelo, st.session_state['dados_previsao'])
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'paper_bgcolor': 'rgba(255, 255, 255, 1)',
        'title': {'text': 'Previsão quantidade de vendas','font':{'color':'black'}},
        'xaxis': {'title': {'text': 'Data','font':{'color':'black'}},'tickfont':{'color':'black'},'gridcolor':'lightgrey'},
        'yaxis': {'title': {'text': 'Vendas','font':{'color':'black'}},'tickfont':{'color':'black'},'gridcolor':'lightgrey'},
    }) 
    st.plotly_chart(fig)
    previsao = st.session_state['dados_previsao']
    
    tabela_previsao = previsao[['ds', 'yhat']].tail(dias)
    tabela_previsao = tabela_previsao.rename(columns={'ds': 'Data', 'yhat': 'Previsão de Vendas'})
    tabela_previsao['Data (Dia/Mês/Ano)'] = tabela_previsao['Data'].dt.strftime('%d/%m/%Y')
    tabela_previsao['Previsão de Vendas'] = tabela_previsao['Previsão de Vendas'].round(2)
    tabela_previsao.reset_index(drop=True, inplace=True)
    st.write('Tabela contendo as previsões de vendas para os próximos {} dias:'.format(dias))
    st.dataframe(tabela_previsao, height=300)

    csv = tabela_previsao.to_csv(index=False)
    st.download_button(label='Baixar tabela como CSV',
                      data=csv,
                      file_name='previsao_de_vendas.csv',
                      mime='text/csv')