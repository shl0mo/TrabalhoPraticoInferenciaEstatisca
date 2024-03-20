import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import requests
import zipfile
import matplotlib.pyplot as plt

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
    tests_options.append("Correlation Analysis")
    tests_options.append("Linear Regression Analysis")
    
    selected_test = st.selectbox("Select a test based on the variable types:", tests_options)
    
    # Execute the selected test
    if selected_test == "Chi-square Test":
        var1,var2= select_colums_categorico(data)
        chi_square_value, p_value, dof, expected = chi_square_test(data,var1,var2)
        print("Chi-square Test Results")
        print( p_value)
        st.write("Chi-square Test Results")
        st.write(f"Chi-square Value: {chi_square_value}")
        st.write(f"P-value: {p_value}")
        st.write(f"Degrees of Freedom: {dof}")

        # Formatar os valores esperados (expected) como um DataFrame para exibição
        expected_df = pd.DataFrame(expected, columns=[f"Var{i+1}" for i in range(expected.shape[1])])
        st.write("Expected Frequencies:")
        st.dataframe(expected_df)

        
    elif selected_test == "T-test":
        select_colums_categorico(data)
        results = student_t_test(data)
        display_test_results("T-test Results", results)
        
    elif selected_test == "ANOVA":
        select_colums_categorico(data)
        results = anova_test(data)
        display_test_results("ANOVA Results", results)
    
    if selected_test == "Correlation Analysis":    
        var1, var2 = select_columns_continuo(data)
        results = correlation_analysis(data, var1, var2)
        display_test_results("Correlation Analysis Results", results)
    elif selected_test == "Linear Regression Analysis":
        var1, var2 = select_columns_continuo(data)
        results = linear_regression_analysis(data, var1, var2)
        display_test_results("Linear Regression Analysis Results", results)
        
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

def display_test_results(title, results):
    st.subheader(title)
    if isinstance(results, list):
        # Assuming results are in the format: [statistic, p-value]
        st.metric(label="Statistic", value=f"{results[0]:.3f}")
        st.metric(label="P-value", value=f"{results[1]:.3f}")
    elif isinstance(results, dict):
        for key, value in results.items():
            st.metric(label=key, value=f"{value:.3f}")

def chi_square_test(data, var1, var2):
    # Create contingency table
    contingency_table = pd.crosstab(data[var1], data[var2])
    
    # Convert index and columns to string for compatibility
    contingency_table.index = contingency_table.index.map(str)
    contingency_table.columns = contingency_table.columns.map(str)
    
    # Display the contingency table
    print("Contingency Table:")
    print(contingency_table)
    
   
    contingency_table_df = pd.DataFrame(contingency_table).reset_index()
   

    # Convertendo a tabela de contingência para HTML
    contingency_table_html = contingency_table.to_html()
    
    # Usando st.markdown para exibir o HTML com a opção unsafe_allow_html ativada
    st.markdown("Contingency Table:", unsafe_allow_html=True)
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

def student_t_test(data):
    #rafael
    print("ok")
    # Retorna os resultados dos testes
    test_results={}
    return test_results

def anova_test(data, continuous_var, categorical_var, alpha=0.05):

    if categorical_var not in data.columns or continuous_var not in data.columns:
        raise ValueError("Specified variables must be in the provided DataFrame.")
    
    # Get unique group labels
    group_labels = data[categorical_var].unique()
    
    if len(group_labels) == 2:
        # Perform t-test for two categories
        group1_label, group2_label = group_labels
        results = student_t_test(data[continuous_var], data[categorical_var], group1_label, group2_label)
        display_test_results("T-test Results", results)
    else:
        # Perform ANOVA test for more than two categories
        samples = [data[data[categorical_var] == label][continuous_var] for label in group_labels]
        stat, p_value = stats.f_oneway(*samples)
        results = [stat, p_value]
        display_test_results("ANOVA Results", results)
              
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

    return [correlation_coef, p_value]

def linear_regression_analysis(data, independent_var, dependent_var):

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

    return [model.coef_[0], model.intercept_, model.score(X, y)]

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
    if file_path.endswith('.csv') or file_path.endswith('.data'):
        data = pd.read_csv(file_path, header=None)
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