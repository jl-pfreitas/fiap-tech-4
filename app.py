import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Carregando os dados 
dados = pd.read_csv('base_ipea_2025_05_01.csv', sep=';', parse_dates=[0], decimal=',', thousands='.')
dados['ano'] = dados['Data'].dt.year

# Configuração inicial do Streamlit
st.set_page_config(
    page_title = 'Previsão do preço (US$) bruto do barril de petróleo tipo Brent',
    layout='wide'
)

# Título Geral
st.markdown("<h1 style='text-align: center; '> Preço (US$) bruto do barril de petróleo tipo Brent</h1>", unsafe_allow_html = True)

# Definindo as páginas
st.sidebar.image('Logo.png', width=200)
paginas = ["Relatório", "Modelo de previsão", 'Dashboard']
pagina_selecionada = st.sidebar.selectbox("Escolha uma página:", paginas)

# Conteúdo da página relatório
if pagina_selecionada == "Relatório":

    # Conteúdo do relatório
    st.markdown('# Relatório Técnico - Análise e Previsão do Preço do Petróleo Brent')
    
    # Capitulo 1
    st.markdown("## 1. Introdução")
    st.markdown('Este relatório tem como objetivo apresentar uma análise exploratória dos preços históricos do petróleo Brent, destacando eventos geopolíticos e econômicos que influenciaram fortemente sua cotação ao longo dos anos. Além disso, é apresentado um modelo de previsão (forecasting) baseado em técnicas de Machine Learning para auxiliar na tomada de decisão estratégica.\n\nA base de dados utilizada foi extraída do site do IPEA, contendo registros diários de preços do petróleo Brent em dólares. A análise foi complementada com um dashboard interativo para facilitar a visualização e extração de insights.')
    
    # Capitulo 2
    st.markdown("## 2. Análise Histórica de Picos Relevantes")
    st.markdown('Ao longo da série histórica, identificamos quatro momentos em que os preços do petróleo sofreram picos acentuados, fortemente associados a eventos geopolíticos e macroeconômicos:')
    
    st.markdown("### 2.1. Guerra do Golfo (1990-1991)")
    st.markdown('Durante a invasão do Kuwait pelo Iraque em agosto de 1990, houve um choque de oferta no mercado global de petróleo. O medo de uma interrupção prolongada na produção elevou drasticamente os preços. Este evento culminou na Primeira Guerra do Golfo, causando incerteza nos mercados e volatilidade nos preços.')
    dados_rel = dados
    dados_rel['Data'] = pd.to_datetime(dados_rel['Data'])
    dados_rel1 = dados_rel[dados_rel['Data'].dt.year.isin([1989, 1990, 1991, 1992])]
    fig_box_rel1 = px.box(dados_rel1, x='ano', y='Preço - petróleo bruto - Brent (FOB)')
    st.plotly_chart(fig_box_rel1)
    
    st.markdown("### 2.2. Crise Financeira Global (2008)")
    st.markdown('No segundo semestre de 2008, o preço do petróleo atingiu níveis historicamente altos. Contudo, com o colapso de grandes instituições financeiras e o subsequente esfriamento da economia global, os preços despencaram em poucos meses. Este movimento reflete o impacto direto da retração da demanda global.')
    dados_rel2 = dados_rel[dados_rel['Data'].dt.year.isin([2007, 2008, 2009])]
    fig_box_rel2 = px.box(dados_rel2, x='ano', y='Preço - petróleo bruto - Brent (FOB)')
    st.plotly_chart(fig_box_rel2)
    
    st.markdown("### 2.3. Pandemia da COVID-19 (2020)")
    st.markdown('Com a disseminação global do coronavírus, a demanda por petróleo caiu drasticamente devido às restrições de mobilidade e desaceleração industrial. Em abril de 2020, contratos futuros de petróleo chegaram a valores negativos, algo inédito na história. A recuperação parcial dos preços entre 2021 e 2022 está associada à retomada econômica e ao corte de produção pelos países da OPEP+ (Organização dos Países Exportadores de Petróleo e aliados).')
    dados_rel3 = dados_rel[dados_rel['Data'].dt.year.isin([2019, 2020, 2021])]
    fig_box_rel3 = px.box(dados_rel3, x='ano', y='Preço - petróleo bruto - Brent (FOB)')
    st.plotly_chart(fig_box_rel3)
    
    st.markdown("### 2.4. Guerra na Ucrânia (2022)")
    st.markdown('A invasão da Ucrânia pela Rússia em fevereiro de 2022 causou grande instabilidade nos mercados globais de energia. Como a Rússia é um dos principais exportadores de petróleo, as sanções econômicas impostas ao país impactaram diretamente a oferta global.')
    dados_rel4 = dados_rel[dados_rel['Data'].dt.year.isin([2021, 2022, 2023])]
    fig_box_rel4 = px.box(dados_rel4, x='ano', y='Preço - petróleo bruto - Brent (FOB)')
    st.plotly_chart(fig_box_rel4)
    
    # Capitulo 3
    st.markdown("## 3. Tendências de Longo Prazo")
    st.markdown('Além dos eventos pontuais, o mercado de petróleo é fortemente influenciado por variáveis estruturais que podem moldar sua trajetória futura:')
    
    st.markdown("### 3.1. Política Comercial dos EUA")
    st.markdown('A adoção de tarifas e sanções econômicas, principalmente em relação ao Irã e Venezuela, tem impacto direto na oferta global. O reposicionamento dos EUA como exportador de petróleo de xisto também influencia a dinâmica global de preços.')
    
    st.markdown("### 3.2. Conflitos Geopolíticos Atuais")
    st.markdown('A continuidade da guerra na Ucrânia, somada à escalada do conflito em Gaza, mantêm elevados os riscos de ruptura no fornecimento de energia, sustentando a volatilidade no mercado de petróleo.')
    
    st.markdown("### 3.3. Crise na Venezuela")
    st.markdown('Com grande potencial de produção, a instabilidade política e econômica na Venezuela, e as sanções internacionais, impede que o país retome sua capacidade plena de exportação, afetando a oferta global.')
    
    # Capitulo 4
    st.markdown("## 4. Forecasting com Machine Learning")
    
    st.markdown("### 4.1. Metodologia")
    st.markdown('Para treinar o nosso modelo, seguimos o passo a passo abaixo:\n* Análise inicial e tratamento dos dados;\n* Identificação de sazonalidade, tendências e ruídos;\n* Transformação logarítmica e de diferenciação para tornar nossa série estacionária;\n* Testes ACF e PACF;\n* Termos de Fourrier;\n* Treinamento dos modelos.')
    
    st.markdown("### 4.2. Resultados")
    st.markdown('| Modelo | WMAPE | Acuracidade |\n|-|-|-|\n| ARIMA + Backtesting | 0.68% | 99.32% |\n| Naive | 6.70% | 93.30% |\n| ARIMA | 6.76% | 93.24% |\n| Seasonal Window Average| 9.75% | 90.25% |\n| Seasonal Naive | 9.95% | 90.05% |\n\nComo podemos ver, o modelo ARIMA + Backtesting apresentou os melhores resultados, porém um ponto negativo desse modelo, é a curta janela de previsão, ideal para um ou dois dias apenas, além de necessitar ser retreinado diariamente com os dados atuais.')
    
    # Capitulo 5
    st.markdown("## 5. Conclusão e Recomendação")
    st.markdown('A análise histórica demonstra que os preços do petróleo Brent são altamente sensíveis a eventos geopolíticos e econômicos globais. Embora o modelo de Machine Learning escolhido ofereça boas previsões para curto prazo, recomendamos que o cliente utilize o dashboard interativo junto à análise contextual dos eventos mundiais como suporte fundamental para decisões estratégicas.\n\nPara maior robustez, também é essencial que o modelo seja alimentado diariamente com os dados atualizados, e o acompanhamento dos eventos geopolíticos que afetem direta e indiretamente a demanda pelo patróleo global.')

# Conteúdo da página modelo
elif pagina_selecionada == "Modelo de previsão":

    # Carregar o modelo salvo
    model = joblib.load('model_arima.joblib')

    # Título da página
    st.title('Previsão com o modelo ARIMA + Backtesting')

    # Entrada de dados para a previsão (ajuste conforme necessário)
    input_data = st.number_input('Insira quantos dias você gostaria de prever:', min_value=0, step=1, format="%d")

    # Função para fazer a previsão
    def fazer_previsao(model, h):
        return model.predict(h=h)

    # Botão para enviar e fazer a previsão
    if st.button('Prever'):
        try:

            # Previsão
            resultado = fazer_previsao(model, h=input_data)

            # Formatação do resultado
            resultado = resultado.rename(columns={'ds': 'Data', 'AutoARIMA': 'Preço (US$) - Petróleo Bruto - Brent (FOB)'})
            resultado['Data'] = resultado['Data'].dt.strftime('%d/%m/%Y')
            resultado['Preço (US$) - Petróleo Bruto - Brent (FOB)'] = resultado['Preço (US$) - Petróleo Bruto - Brent (FOB)'].round(2)
            resultado.set_index('Data', inplace=True)

            # Exibição do resultado
            st.title(f'Previsão dos próximos {input_data} dias')
            st.subheader("Tabela de Preços")
            st.dataframe(resultado)
        except ValueError:
            st.error('Valor inválido, favor inserir um número inteiro positivo')

    # Dados de referência
    st.title('Preço dos últimos 5 dias úteis')
    dados_referencia = dados.iloc[1:]
    dados_referencia['Data'] = dados_referencia['Data'].dt.strftime('%d/%m/%Y')
    dados_referencia.set_index('Data', inplace=True)
    dados_referencia = dados_referencia.drop(columns=['ano'])
    st.dataframe(dados_referencia.head())


# Conteúdo da página Dashboard
elif pagina_selecionada == "Dashboard":

    # Layout em colunas para os filtros de data
    col_data1, col_data2 = st.columns(2)

    with col_data1:
        data_inicial = st.date_input(
            'Data Inicial',
            dados['Data'].min().date(),
            min_value=dados['Data'].min().date(),
            max_value=dados['Data'].max().date()
        )

    with col_data2:
        data_final = st.date_input(
            "Data Final",
            dados['Data'].max().date(),
            min_value=dados['Data'].min().date(),
            max_value=dados['Data'].max().date()
        )

    # Filtrar os dados com base nas datas
    dados['Data'] = pd.to_datetime(dados['Data'])
    filtro_data = (dados['Data'] >= pd.to_datetime(data_inicial)) & (dados['Data'] <= pd.to_datetime(data_final))
    dados_filtrados = dados.loc[filtro_data]

    # Define o preço máximo e mínimo do petróleo no intervalo filtrado
    preco_max = dados_filtrados['Preço - petróleo bruto - Brent (FOB)'].max()
    preco_min = dados_filtrados['Preço - petróleo bruto - Brent (FOB)'].min()

    # Converter para datas
    primeira_data = dados_filtrados['Data'].min().strftime('%d/%m/%Y')
    ultima_data = dados_filtrados['Data'].max().strftime('%d/%m/%Y')

    # Primeiro e último registros completos
    primeiro_registro = dados_filtrados.sort_values('Data').iloc[0]
    ultimo_registro = dados_filtrados.sort_values('Data').iloc[-1]

    #Primeiro e último valor registrado
    valor_primeiro = primeiro_registro['Preço - petróleo bruto - Brent (FOB)']
    valor_ultimo = ultimo_registro['Preço - petróleo bruto - Brent (FOB)']

    #Cards 

        # Estilo "vidro fosco"
    
    st.markdown("""
        <style>
        .card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        text-align: center;
        font-family: sans-serif;
        color: #ffffff;
    }
    .card h3 {
        margin: 0;
        font-size: 20px;
    }
    .card p {
        margin: 5px 0 0;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
    
    #Layout com os 4 cards

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class="card">
                <h3>📈 Preço Máximo</h3>
                <span>US$ {preco_max:,.2f}</span>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="card">
                <h3>📉 Preço Mínimo</h3>
                <span>US$ {preco_min:,.2f}</span>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="card">
                <h3>🕐 Primeiro Registro</h3>
                <span>US$ {valor_primeiro:,.2f}</span>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="card">
                <h3>🕔 Último Registro</h3>
                <span>US$ {valor_ultimo:,.2f}</span>
            </div>
        """, unsafe_allow_html=True)
            

    # Gráfico de linhas
    fig_linha = px.line(dados_filtrados, x='Data', y='Preço - petróleo bruto - Brent (FOB)', title='Preço (US$) - Petróleo Bruto - Brent (FOB) ao longo do Tempo')
    st.plotly_chart(fig_linha)

    # Boxplot
    fig_box = px.box(dados_filtrados, x='ano', y='Preço - petróleo bruto - Brent (FOB)', title='Boxplot do preço (US$) - Petróleo Bruto - Brent (FOB) por ano')
    st.plotly_chart(fig_box)

    # Calcula a média anual com base nos dados filtrados
    media_anual = dados_filtrados.groupby('ano')['Preço - petróleo bruto - Brent (FOB)'].mean().reset_index()
    
    # Cria o gráfico de barras
    fig_media_anual = px.bar(
    media_anual,
    x='ano',
    y='Preço - petróleo bruto - Brent (FOB)',
    title='Média Anual do Preço (US$) - Petróleo Bruto - Brent (FOB)',
    labels={'ano': 'Ano', 'Preço - petróleo bruto - Brent (FOB)': 'Preço Médio (US$)'},
    text_auto='.2f'
    )

    # Exibe o gráfico
    st.plotly_chart(fig_media_anual)