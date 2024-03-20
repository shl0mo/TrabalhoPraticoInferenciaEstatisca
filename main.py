import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import requests
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
import unicodedata
import re

import streamlit as st

from sklearn.linear_model import LinearRegression

def getTestDone(data):
    # Preliminary checks and setup
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    st.session_state['data'] = data
   
    tests_options = []
    # Scenario handling based on variable types
    
    tests_options.append("Chi-square Test")
    tests_options.append("T-test")
    tests_options.append("ANOVA")
    tests_options.append("Mann-Whitney")
    tests_options.append("Correlation Analysis")
    tests_options.append("Linear Regression Analysis")
    
    selected_test = st.selectbox("Select a test based on the variable types:", tests_options)
    
    # Execute the selected test
    if selected_test == "Chi-square Test":
        var1,var2= select_colums_categorico(data)
        chi_square_value, p_value, dof, expected = chi_square_test(data,var1,var2)
        
        with st.expander("Resultados do Teste Qui-Quadrado"):
            st.markdown("""
            ### Resultados do Teste Qui-Quadrado

            - **Valor de Qui-Quadrado:** {:.4f}
            - **p-valor:** {:.4f}
            - **Graus de Liberdade:** {}

            Dependendo do p-valor, podemos determinar se rejeitamos ou não a hipótese nula. Um p-valor baixo (por exemplo, menor que 0.05) indica que há evidências suficientes para rejeitar a hipótese nula, sugerindo que as variáveis analisadas estão associadas.
            """.format(chi_square_value, p_value, dof))

        # Formatar os valores esperados (expected) como um DataFrame para exibição
        expected_df = pd.DataFrame(expected, columns=[f"Var{i+1}" for i in range(expected.shape[1])])
        st.write("Expected Frequencies:")
        st.dataframe(expected_df)
        
    elif selected_test == "T-test":
        var1, var2 = select_columns_continuo(data)
        stat, p_value = student_t_test(data,var1, var2)
        with st.expander("Resultados do Teste t 📘"):
            st.markdown("""
            ### Resultados do Teste t 📈

            - **Estatística t:** {:.3f}
            - **p-valor:** {:.3f}

            Dependendo do p-valor, podemos interpretar a significância dos resultados:
            - Um **p-valor baixo** (por exemplo, < 0.05) 📉 sugere que há evidências suficientes para rejeitar a hipótese nula, indicando uma diferença significativa entre os grupos.
            - Um **p-valor alto** (por exemplo, ≥ 0.05) 📊 sugere que não há evidências suficientes para rejeitar a hipótese nula, indicando que as diferenças entre os grupos podem não ser significativas.
            """.format(stat, p_value))

           
    elif selected_test == "ANOVA":
        var1,var2 =select_one_column_continuo_and_the_other_categorical(data)
        results = anova_test(data,var1,var2)

    elif selected_test == "Mann-Whitney":
        mann_whitney_test(data)
            
    if selected_test == "Correlation Analysis":    
        var1, var2 = select_columns_continuo(data)
        correlation_coef, p_value = correlation_analysis(data, var1, var2)
        with st.expander("Resultados da Análise de Correlação 📘"):
            st.markdown("""
            ### Resultados da Análise de Correlação 📊

            - **Coeficiente de Correlação:** {:.3f}
            - **p-valor:** {:.3f}

            A interpretação do coeficiente de correlação é a seguinte:
            - Um valor **próximo de 1** ou **-1** indica uma correlação forte 🚀. Um valor positivo sugere uma correlação positiva, enquanto um valor negativo indica uma correlação negativa.
            - Um valor **próximo de 0** indica que não há correlação linear significativa entre as variáveis 🛤.

            O p-valor ajuda a determinar a significância estatística da correlação observada:
            - Um **p-valor baixo** (por exemplo, < 0.05) 📉 sugere que a correlação é estatisticamente significativa, indicando que é improvável que a correlação observada tenha ocorrido por acaso.
            - Um **p-valor alto** (por exemplo, ≥ 0.05) 📊 sugere que a correlação pode não ser estatisticamente significativa.
            """.format(correlation_coef, p_value))

    elif selected_test == "Linear Regression Analysis":
        var1, var2 = select_columns_continuo(data)
        modelcoef, modelintercept, modelscore = linear_regression_analysis(data, var1, var2)
        with st.expander("Resultados da Análise de Regressão Linear 📘"):
            st.markdown("""
            ### Resultados da Análise de Regressão Linear 📈

            - **Coeficiente do Modelo:** {:.3f}
            - **Intercepto do Modelo:** {:.3f}
            - **Score do Modelo (R²):** {:.3f}

            🧐 **Interpretação:**
            - O **coeficiente** indica a mudança esperada na variável dependente para uma unidade de mudança na variável independente.
            - O **intercepto** representa o valor esperado da variável dependente quando a variável independente é 0.
            - O **score do modelo (R²)** mede a proporção da variância na variável dependente que é previsível a partir da variável independente(s).
            """.format(modelcoef, modelintercept, modelscore))
        
def select_columns_continuo(data):
    # verify cada coluna do dataset se é numérica ou categórica
    available_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    var1 = st.selectbox('Select the first variable for comparison:', available_columns, index=0)
    var2 = st.selectbox('Select the second variable for comparison:', available_columns, index=min(1, len(available_columns)-1))
    st.session_state['var1'] = var1
    st.session_state['var2'] = var2
    return var1, var2

def select_colums_categorico(data):
    
    available_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    var1 = st.selectbox('Select the first variable for comparison:', available_columns, index=0)
    var2 = st.selectbox('Select the second variable for comparison:', available_columns, index=min(1, len(available_columns)-1))
    st.session_state['var1'] = var1
    st.session_state['var2'] = var2
    return var1, var2

def select_one_column_continuo_and_the_other_categorical(data):
    # verify cada coluna do dataset se é numérica ou categórica
    available_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    var1 = st.selectbox('Select the continuous variable:', available_columns, index=0)
    available_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    var2 = st.selectbox('Select the categorical variable:', available_columns, index=0)
    st.session_state['var1'] = var1
    st.session_state['var2'] = var2
    return var1, var2

def chi_square_test(data, var1, var2):
    with st.expander("Sobre o Teste Qui-quadrado (Chi-square) 📘"):
        st.markdown("""
        O **Teste Qui-quadrado (Chi-square)** é usado para investigar se existe uma associação entre duas categorias de uma variável. É útil para tabelas de contingência e pode ajudar a determinar se as diferenças entre as categorias são estatisticamente significativas.
        
        - **Hipótese Nula (H0):** Não há associação entre as variáveis categóricas.
        - **Hipótese Alternativa (H1):** Existe uma associação entre as variáveis categóricas.
        
        O teste fornece um valor de Qui-quadrado, um valor P associado e os graus de liberdade da tabela de contingência. Um valor P baixo sugere que devemos rejeitar a hipótese nula, indicando uma associação significativa entre as variáveis.
        """)
    # Create contingency table
    contingency_table = pd.crosstab(data[var1], data[var2])
    
    with st.expander("Visualize o Heatmap 🔥"):
        # Configura o tamanho da figura antes de criar o heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
        # Adiciona títulos ou ajustes finais aqui, se necessário
        plt.title("Tabela de Heatmap")
        # Exibe o plot
        st.pyplot(plt)
        # Limpa a figura após renderização para evitar sobreposições em atualizações subsequentes
        plt.clf()

    
    # Convert index and columns to string for compatibility
    contingency_table.index = contingency_table.index.map(str)
    contingency_table.columns = contingency_table.columns.map(str)

    # Convertendo a tabela de contingência para HTML
    contingency_table_html = contingency_table.to_html()
    
    # Usando st.markdown para exibir o HTML com a opção unsafe_allow_html ativada
    with st.expander("Tabela de Contingência"):
        st.markdown("Contingency Table:")
        st.markdown(contingency_table_html, unsafe_allow_html=True)
    
    # Perform the chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2, p, dof, expected

def generateVisualizationsInTabs(data):
    available_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    with st.container():
        # Abas principais para Histogramas e Box Plots
        tab_histograms, tab_boxplots = st.tabs(["Histograms", "Box Plots"])
        
        # Geração de Histogramas em uma aba principal
        with tab_histograms:
            for column in available_columns:
               
                # Usando st.expander para tornar cada histograma colapsável
                with st.expander(f"Histogram for {column}"):
                    fig, ax = plt.subplots()
                    ax.hist(data[column])
                    print("data column test")
                    print(data[column].head())
                    ax.set_title(f"Histogram for {column}")
                    st.pyplot(fig)
        
        # Geração de Box Plots em outra aba principal
        with tab_boxplots:
            for column in available_columns:
                # Usando st.expander para tornar cada box plot colapsável
                with st.expander(f"Boxplot for {column}"):
                    fig, ax = plt.subplots()
                    ax.boxplot(data[column], vert=False)
                    ax.set_title(f"Boxplot for {column}")
                    st.pyplot(fig)      

def student_t_test(data, var1, var2):
    with st.expander("Sobre o Teste T de Student 📘"):
        st.markdown("""
        O **Teste T de Student** é utilizado para comparar as médias de duas amostras independentes e avaliar se há uma diferença estatisticamente significativa entre elas. Esse teste é útil quando você quer entender se duas condições experimentais resultam em diferentes efeitos médios. Por exemplo, pode ser usado para comparar a eficácia de dois medicamentos.
        
        - **Hipótese Nula (H0):** Não há diferença significativa entre as médias das duas amostras.
        - **Hipótese Alternativa (H1):** Existe uma diferença significativa entre as médias das duas amostras.
        """)
        
    # Selecionando os dados das colunas especificadas pelos índices var1 e var2
    sample1 = data.iloc[:, var1]
    sample2 = data.iloc[:, var2]
    
    # Removendo possíveis NaNs para evitar erros no teste
    sample1 = sample1.dropna()
    sample2 = sample2.dropna()

    # Utilizando o expander para mostrar as amostras de maneira colapsável
    with st.expander("Ver Amostra 1👀"):
        st.table(sample1)

    with st.expander("Ver Amostra 2 👀"):
        st.table(sample2)

    # Realizando o teste t para amostras independentes
    stat, p_value = stats.ttest_ind(sample1, sample2)
    
    return stat, p_value    

def anova_test(data, continuous_var, categorical_var, alpha=0.05):
    with st.expander("Sobre o Teste ANOVA 📘"):
        st.markdown("""
        O **Teste ANOVA (Análise de Variância)** é utilizado para comparar as médias entre três ou mais grupos independentes. Isso é útil para determinar se pelo menos um grupo difere significativamente dos outros. Por exemplo, pode ser utilizado para avaliar se a resposta a três tipos diferentes de tratamento é diferente.

        - **Hipótese Nula (H0):** As médias dos grupos são todas iguais.
        - **Hipótese Alternativa (H1):** Pelo menos uma média de grupo é diferente das outras.
        """)

    # Agrupar os dados pela variável categórica
    groups = data.groupby(categorical_var)[continuous_var].apply(list).tolist()

    # Realizar o teste ANOVA
    f_statistic, p_value = stats.f_oneway(*groups)

    # Preparar e mostrar os resultados
    if p_value < alpha:
        result_text = f"Com um p-valor de {p_value:.4f}, há evidências suficientes para rejeitar a hipótese nula. Portanto, existe uma diferença significativa entre as médias dos grupos. 👌"
    else:
        result_text = f"Com um p-valor de {p_value:.4f}, não há evidências suficientes para rejeitar a hipótese nula. Assim, não podemos afirmar que existe uma diferença significativa entre as médias dos grupos.⛔"

    # Exibição dos resultados
    with st.expander("Resultados do Teste ANOVA"):
        st.markdown("### ANOVA Results")
        st.text(f"F-Statistic: {f_statistic:.4f}\nP-Valor: {p_value:.4f}\n{result_text}")

def mann_whitney_test(data, alpha=0.05):
    with st.expander("Sobre o teste Mann-Whitney 📘"):
        st.markdown("""
            O teste de **Mann-Whitney** é um teste não paramétrico utilizado para determinar se duas amostras independentes foram tiradas de populações com a mesma distribuição. Ele é usado quando as suposições necessárias para o teste T de Student não são atendidas, como quando os dados não são normalmente distribuídos.
            - **Hipótese nula (H0):** As distribuições das duas amostras são iguais.
            - **Hipótese alternativa (H1):** As distribuições das duas amostras não são iguais.
        """)

    tests_df = data[data[4] == "TP"].head(10).reset_index().drop("index", axis=1)
    exercises_df = data[data[4] == "LAB"].head(10).reset_index().drop("index", axis=1)

    with st.expander("Ver Amostra 1 👀"):
        st.table(tests_df)

    with st.expander("Ver Amostra 2 👀"):
        st.table(exercises_df)
    
    tests_group = tests_df[3].tolist()
    exercises_group = exercises_df[3].tolist()
    n1 = len(tests_group)
    n2 = len(exercises_group)
    u_obs, p_value = stats.mannwhitneyu(tests_group, exercises_group, alternative="two-sided")
    result_test = ""
    if (p_value >= alpha):
        result_text = f"Considerando que o valor-p $p$ obtido foi maior ou igual ao nível de significância $\\alpha$, isso é, ${p_value:.4f} >= {alpha}$, então não rejeitamos a hipótese nula de que as amostras possuem distribuições iguais. 👌"
    else:
        result_text = f"Considerando que o valor-p $p$ obtido foi menor que o nível de significância $\\alpha$, isso é, ${p_value:.4f} < {alpha}$ então rejeitamos a hipótese nula de que as amostras possuem distribuições iguais. ⛔"
    with st.expander("Resultados do Teste Mann-Whitney"):
        st.markdown("### Mann-Whitney Results")
        obs_string = "obs"
        st.markdown(f"""
                $n_{1} = {n1}$\n
                $n_{2} = {n2}$\n
                $\\alpha = {alpha}$\n
                $U_{{obs}} = {u_obs:.4f}$\n
                $p = {p_value:.4f}$\n
                **Conclusão**: {result_text}
        """)

              
def identify_variables(data):
    num_vars = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_vars = data.select_dtypes(include=['object', 'category']).columns.tolist()

    categorical_var = None
    continuous_var = None

    if len(num_vars) == 1 and len(cat_vars) == 1:
        continuous_var = num_vars[0]
        categorical_var = cat_vars[0]

    return categorical_var, continuous_var

def correlation_analysis(data, var1, var2):
    with st.expander("Sobre a Análise de Correlação 📘"):
        st.markdown("""
        A **Análise de Correlação** mede a associação entre duas variáveis contínuas, fornecendo um coeficiente de correlação (geralmente de Pearson) que indica a força e a direção dessa associação. Valores próximos de 1 ou -1 indicam uma forte correlação positiva ou negativa, respectivamente, enquanto valores próximos de 0 indicam nenhuma correlação.
        
        - **Coeficiente de Correlação de Pearson:** Mede o grau de relação linear entre duas variáveis.
        - **Valor P:** Testa a hipótese de não haver correlação entre as variáveis. Um valor p baixo (tipicamente ≤ 0.05) indica que você pode rejeitar a hipótese nula de não correlação.
          """)
    if var1 not in data.columns or var2 not in data.columns:
        raise ValueError("Both variables must be present in the data.")

    correlation_coef, p_value = stats.pearsonr(data[var1], data[var2])
    
    print(f"Pearson correlation between {var1} and {var2}: {correlation_coef:.3f}")
    print(f"P-value: {p_value:.3f}")
    
    if p_value < 0.05:
        print("The correlation is statistically significant.")
        st.write("The correlation is statistically significant.")
    else:
        print("The correlation is not statistically significant.")
        st.write("The correlation is not statistically significant.")

    return correlation_coef, p_value

def linear_regression_analysis(data, independent_var, dependent_var):
    with st.expander("Sobre a Análise de Regressão Linear 📘"):
        st.markdown("""
        A **Análise de Regressão Linear** é um método estatístico que modela a relação entre uma variável dependente e uma ou mais variáveis independentes, assumindo uma relação linear entre elas. Este modelo é representado pela equação `y = mx + b`, onde `y` é a variável dependente, `m` é o coeficiente da variável independente (inclinação), `x` é a variável independente, e `b` é o intercepto.

        - **Coeficiente (inclinação):** Indica quanto `y` muda por uma unidade de mudança em `x`.
        - **Intercepto:** O valor de `y` quando `x` é 0.
        - **R-quadrado:** Mede o quanto da variabilidade de `y` pode ser explicada pelo modelo. Valores mais altos indicam um melhor ajuste do modelo aos dados.
        """)
    if independent_var not in data.columns or dependent_var not in data.columns:
        raise ValueError("Both variables must be present in the data.")
    
    X = data[independent_var].values.reshape(-1, 1)  # Independent variable
    y = data[dependent_var].values  # Dependent variable

    model = LinearRegression()
    model.fit(X, y)
    
    print(f"Regression model for {dependent_var} as a function of {independent_var}:")
    print(f"Slope (coefficient): {model.coef_[0]:.3f}")
    print(f"Intercept: {model.intercept_:.3f}")
    print(f"R-squared: {model.score(X, y):.3f}")
    print("Higher R-squared values indicate a better fit for the model.")

    st.write(f"Regression model for {dependent_var} as a function of {independent_var}:")
    st.write(f"Slope (coefficient): {model.coef_[0]:.3f}")
    st.write(f"Intercept: {model.intercept_:.3f}")
    st.write(f"R-squared: {model.score(X, y):.3f}")
    st.write("Higher R-squared values indicate a better fit for the model.")

    return model.coef_[0], model.intercept_, model.score(X, y)

def download_dataset(dataset_url):
    # Check if the dataset exists
    if not verify_if_exists_dataset(dataset_url):
        print('Dataset does not exist')
        return None
    
     # Verifica se o dataset já foi baixado
    if dataset_url in st.session_state:
        print('Dataset já foi baixado anteriormente.')
        return st.session_state[dataset_url]
    
    # Verifica se o dataset existe
    if not verify_if_exists_dataset(dataset_url):
        print('Dataset não existe')
        return None
    

    # Extract the dataset ID and name from the URL
    parts = dataset_url.split('/')
    dataset_name = parts[-1]  # Assuming the last part is the dataset name
    dataset_id = parts[-2]  # Assuming the second last part is the dataset ID

    # Verifica se o dataset já foi baixado
    dataset_path = os.path.join('Data', dataset_name)
    if os.path.exists(dataset_path):
        print('Dataset já foi baixado anteriormente.')
        return dataset_path
    
    # Construct the download URL
    download_url = f"https://archive.ics.uci.edu/static/public/{dataset_id}/{dataset_name}.zip"
    print(f'Download URL: {download_url}')
    # retire todos os caracteres epecias  e espeacos
    normalized_dataset_name = re.sub('[\n\r\t]+', '', dataset_name)  # Remove espaços, quebras de linha, e tabs
    normalized_dataset_name = re.sub('[\+\-]+', '', normalized_dataset_name)  # Remove símbolos específicos como + e -

    # Para remover acentos e normalizar caracteres
    normalized_dataset_name = ''.join((c for c in unicodedata.normalize('NFD', normalized_dataset_name) if unicodedata.category(c) != 'Mn'))

    
    # Define download_path outside of the if statement to ensure it's always available
    download_path = os.path.join('Data', dataset_name + '.zip')

    # Download the file
    response = requests.get(download_url, stream=True)

    if response.status_code == 200:
        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print('Download complete!')
    else:
        print(f'Download failed with status code: {response.status_code}')
        return None  # Return or handle error appropriately

    # Path to save the dataset same as the zip file but without '.zip'
    dataset_path_to_be_saved = os.path.join('Data', dataset_name)

    # Unzip the dataset
    dataset_path = unzip_dataset(download_path, dataset_path_to_be_saved,dataset_name)

     # Atualiza o st.session_state com o caminho do dataset
    st.session_state[dataset_url] = dataset_path

    # Delete zip file to clean up
    os.remove(download_path)

    # Return dataset_path unzipped
    return dataset_path
 
def verify_if_exists_dataset(dataset_url):
    # ping the URL
    response = requests.get(dataset_url)
    if response.status_code == 200:
        print(f'Dataset exists at {dataset_url}')
    else:
        print(f'Dataset does not exist at {dataset_url}')

    return response.status_code == 200

def unzip_dataset(path_to_zip_file, directory_to_extract_to,dataset_name):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    print('Unzip complete!')

    # dataset path name_of_the_data_set.data
    return os.path.join(directory_to_extract_to)

def load_dataset(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path, header=None,sep=';')
    elif file_path.endswith('.data'):
        data = pd.read_csv(file_path, header=None)
        if "codebench" in file_path:
            data[1] = data[1].astype("str")
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path, header=None)
    else:
        raise ValueError("Unsupported file format.")
    
    return data

def get_dataset_info(dataset_path):

    data = load_dataset(dataset_path)
    
    
    return data.head(), data.shape, data.columns, data.dtypes, data.isnull().sum(), data.describe(), data

def getDataSetFromPath(dataset_path):
    # Lista para armazenar os caminhos completos dos arquivos .data
    datafiles_path = []
    # Percorre o diretório especificado e seus subdiretórios
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".data"):
                # Concatena o diretório raiz com o nome do arquivo para formar o caminho completo
                full_path = os.path.join(root, file)
                datafiles_path.append(full_path)
            if file.endswith(".xlsx"):
                # Concatena o diretório raiz com o nome do arquivo para formar o caminho completo
                full_path = os.path.join(root, file)
                datafiles_path.append(full_path)
            if file.endswith(".csv"):
                full_path = os.path.join(root, file)
                datafiles_path.append(full_path)
            
    return datafiles_path

def app():
    st.title('📊 Análise de Datasets')

    # Input do usuário para o link do dataset
    dataset_url = st.text_input('🔗 Insira o link do dataset UCI', 'https://archive.ics.uci.edu/dataset/2/adult')
    
    if st.button('📥 Baixar Dataset'):
        with st.spinner('Baixando o dataset...'):
            dataset_path = download_dataset(dataset_url)
            if dataset_path is None:
                st.error('❌ Dataset não encontrado ou erro no download!')
            else:
                st.success('✅ Dataset baixado com sucesso!')
                st.info(f'📁 Path do dataset: `{dataset_path}`')
                # Store the dataset path in the session state
                st.session_state['dataset_path_complete'] = dataset_path

    # This part assumes dataset_path_complete is correctly set to the path where the dataset files are.
    if 'dataset_path_complete' in st.session_state:
        all_data = getDataSetFromPath(st.session_state['dataset_path_complete'])
       
        if(len(all_data) == 0):
            st.write('Arquivos do dataset:', 'Nenhum arquivo encontrado')
            return
        
        # Make the user choose the file
        selected_file = st.selectbox('Selecione o arquivo do dataset', all_data)
        # Load the dataset
        data_head, data_shape, data_columns, data_dtypes, data_missing_values, data_description , selected_dataset = get_dataset_info(selected_file)
        
        if st.checkbox('🔍 Visualizar informações do dataset'):
            # Unpack the returned values from get_dataset_info
            
            
            # Create a tab layout for organizing the information
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Visão Geral", 
                "Primeiras Linhas", 
                "Colunas e Tipos", 
                "Valores Ausentes", 
                "Estatísticas",
                "Observações"
            ])
            
            with tab1:
                st.write('Formato do dataset:', data_shape)
                st.write('Total de Colunas:', len(data_columns))
                st.write('Total de Valores Ausentes:', data_missing_values.sum())
            
            with tab2:
                st.write('Primeiras 5 linhas do dataset:')
                st.dataframe(data_head)
            
            with tab3:
                st.write('Colunas do dataset e seus tipos de dados:')
                # Combine column names and types into a DataFrame for better display
                cols_types_df = pd.DataFrame({
                    'Coluna': data_columns,
                    'Tipo de Dado': data_dtypes.values
                })
                st.table(cols_types_df)
            
            with tab4:
                st.write('Valores ausentes nas colunas:')
                # Convert the Series of missing values into a DataFrame for better display
                missing_values_df = pd.DataFrame(data_missing_values, columns=['Valores Ausentes'])
                missing_values_df = missing_values_df[missing_values_df['Valores Ausentes'] > 0]  # Filter columns with missing values
                st.table(missing_values_df)
            
            with tab5:
                st.write('Estatísticas do dataset:')
                st.dataframe(data_description)
            
            with tab6:
                st.write('Observações:')
                st.markdown("""
                - **Primeiras Linhas**: Uma rápida visão das primeiras entradas do dataset.
                - **Colunas e Tipos**: Detalhamento das colunas disponíveis e seus tipos de dados.
                - **Valores Ausentes**: Visão dos dados ausentes que podem requerer tratamento.
                - **Estatísticas**: Sumário estatístico do dataset para uma análise inicial.
                """)
                
        # Let the user select two variables from the dataset
        if selected_dataset is not None and not selected_dataset.empty:
            # show histogram and boxplot
            generateVisualizationsInTabs(selected_dataset)
            
            # Assuming 'selected_file' is the file selected by the user
            if 'dataset_path_complete' in st.session_state :
                # Call the modified function with the selected dataset
                getTestDone(selected_dataset)
                
if __name__ == "__main__":
    app()
