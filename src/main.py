import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import requests
import zipfile

import streamlit as st

from sklearn.linear_model import LinearRegression

def getTestDone(data, var1, var2):
    # Preliminary checks and setup
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    st.session_state['data'] = data
    st.session_state['var1'] = var1
    st.session_state['var2'] = var2

    # Determine variable types
    is_var1_categorical = pd.api.types.is_categorical_dtype(data[var1]) or pd.api.types.is_object_dtype(data[var1])
    is_var2_categorical = pd.api.types.is_categorical_dtype(data[var2]) or pd.api.types.is_object_dtype(data[var2])

    tests_options = []
    # Scenario handling based on variable types
    if is_var1_categorical and is_var2_categorical:
        tests_options.append("Chi-square Test")
    elif is_var1_categorical != is_var2_categorical:
        tests_options.append("T-test")
        if len(data[var1].unique()) > 2 or len(data[var2].unique()) > 2:
            tests_options.append("ANOVA")
    else:
        tests_options.append("Correlation Analysis")
        tests_options.append("Linear Regression Analysis")

    selected_test = st.selectbox("Select a test based on the variable types:", tests_options)

    # Execute the selected test
    if selected_test == "Chi-square Test":
        st.write("Both variables are categorical. Performing Chi-square Test...")
        # chi_square_test function call goes here - Note: You need to adapt it for two categorical vars or choose another approach
    elif selected_test == "T-test":
        st.write("One variable is categorical with two categories, and the other is continuous. Performing T-test...")
        continuous_var = var1 if not is_var1_categorical else var2
        categorical_var = var1 if is_var1_categorical else var2
        group1_label, group2_label = data[categorical_var].unique()[:2]
        results = student_t_test(data[continuous_var], data[categorical_var], group1_label, group2_label)
        display_test_results("T-test Results", results)
    elif selected_test == "ANOVA":
        st.write("Categorical variable has more than two categories. Performing ANOVA...")
        continuous_var = var1 if not is_var1_categorical else var2
        categorical_var = var1 if is_var1_categorical else var2
        results = anova_test(data, continuous_var, categorical_var)
        display_test_results("ANOVA Results", results)
    elif selected_test == "Correlation Analysis":
        st.write("Both variables are continuous. Performing Correlation Analysis...")
        results = correlation_analysis(data, var1, var2)
        display_test_results("Correlation Analysis Results", results)
    elif selected_test == "Linear Regression Analysis":
        st.write("Both variables are continuous. Performing Linear Regression Analysis...")
        results = linear_regression_analysis(data, var1, var2)
        display_test_results("Linear Regression Analysis Results", results)

def display_test_results(title, results):
    """
    Displays the results of statistical tests in Streamlit.
    
    Parameters:
    - title: Title of the test results to display.
    - results: Results of the test in a list or dictionary format.
    """
    st.subheader(title)
    if isinstance(results, list):
        # Assuming results are in the format: [statistic, p-value]
        st.metric(label="Statistic", value=f"{results[0]:.3f}")
        st.metric(label="P-value", value=f"{results[1]:.3f}")
    elif isinstance(results, dict):
        for key, value in results.items():
            st.metric(label=key, value=f"{value:.3f}")




        
         
def chi_square_test(data, feature_name, alpha=0.05):
    """
    Performs a chi-square test on a specified feature to check if it follows a uniform distribution.

    Parameters:
    - data: DataFrame or similar structure containing the data.
    - feature_name: String name of the feature to be tested.
    - alpha: Significance level for the test. Default is 0.05.

    Returns:
    None, but prints the test results.
    """
    # Extract the feature data
    feature = data[feature_name]
    # Calculate observed values
    observed_values, bins = np.histogram(feature, bins=10)
    # Calculate expected values assuming uniform distribution
    expected_values = np.ones_like(observed_values) * len(feature) / 10
    # Perform chi-square test
    chi_square_statistic, p_value = stats.chisquare(f_obs=observed_values, f_exp=expected_values)
    
    # Print test results
    print(f"Chi-Square Test for the feature '{feature_name}'")
    print(f"Test Statistic: {chi_square_statistic}")
    print(f"P-value: {p_value}")
    if p_value < alpha:
        print("Null hypothesis rejected - The distribution is not uniform.")
        st.write("Null hypothesis rejected - The distribution is not uniform.")
    else:
        print("Failed to reject the null hypothesis - The distribution is uniform.")
        st.write("Failed to reject the null hypothesis - The distribution is uniform.")

    return [chi_square_statistic, p_value]

def student_t_test(feature_data, labels, group1_label, group2_label, alpha=0.05):
    """
    Performs a Student's t-test between two specified groups for a given feature.

    Parameters:
    - feature_data: Array-like, the data of the feature to test.
    - labels: Array-like, the labels corresponding to the feature data.
    - group1_label: The label of the first group for comparison.
    - group2_label: The label of the second group for comparison.
    - alpha: Significance level for the test. Default is 0.05.

    Returns:
    None, but prints the test results.
    """
    # Split the data into two samples based on the labels
    sample1 = feature_data[labels == group1_label]
    sample2 = feature_data[labels == group2_label]
    # Perform Student's t-test
    stat, p_value = stats.ttest_ind(sample1, sample2)

    
    # Print test results
    print(f"Student's t-test for the feature between '{group1_label}' and '{group2_label}'")
    print(f"Test Statistic: {stat}")
    print(f"P-value: {p_value}")
    if p_value < alpha:
        print("Null hypothesis rejected - There is a significant difference between the groups.")
        st.write("Null hypothesis rejected - There is a significant difference between the groups.")
    else:
        print("Failed to reject the null hypothesis - No significant difference between the groups.")
        st.write("Failed to reject the null hypothesis - No significant difference between the groups.")
    print("\n")

    return [stat, p_value]

def anova_test(data, continuous_var, categorical_var, alpha=0.05):
    """
    Performs an ANOVA test to compare means across multiple groups in a single continuous variable.
    
    Parameters:
    - data: DataFrame containing the dataset.
    - continuous_var: String, name of the continuous variable column in `data`.
    - categorical_var: String, name of the categorical variable column in `data`.
    - alpha: Significance level for the test. Default is 0.05.
    
    Returns:
    None, but prints the test results.
    """
    # Ensure categorical_var is in data
    if categorical_var not in data.columns or continuous_var not in data.columns:
        raise ValueError("Specified variables must be in the provided DataFrame.")
    
    # Get unique group labels
    group_labels = data[categorical_var].unique()
    
    # Prepare samples
    samples = [data[data[categorical_var] == label][continuous_var] for label in group_labels]
    
    # Perform ANOVA test
    stat, p_value = stats.f_oneway(*samples)
    
    # Print test results
    print(f"ANOVA test for {continuous_var} across groups in {categorical_var}")
    print(f"Test Statistic: {stat}")
    print(f"P-value: {p_value}")
    if p_value < alpha:
        print("Null hypothesis rejected - There is a significant difference between at least two groups.")
        st.write("Null hypothesis rejected - There is a significant difference between at least two groups.")
    else:
        print("Failed to reject the null hypothesis - No significant difference between the groups.")
        st.write("Failed to reject the null hypothesis - No significant difference between the groups.")
    return [stat, p_value]


def correlation_analysis(data, var1, var2):
    """
    Performs a Pearson correlation analysis between two continuous variables.

    Parameters:
    - data: DataFrame containing the dataset.
    - var1: String, name of the first continuous variable column in `data`.
    - var2: String, name of the second continuous variable column in `data`.

    Prints the correlation coefficient and the p-value.
    """
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
    """
    Performs a simple linear regression analysis between one independent and one dependent variable.

    Parameters:
    - data: DataFrame containing the dataset.
    - independent_var: String, name of the independent variable column in `data`.
    - dependent_var: String, name of the dependent variable column in `data`.

    Prints the slope, intercept, and R-squared value of the regression model.
    """
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


def get_dataset_info(dataset_path_complete):

    # Load the dataset
    data = pd.read_csv(dataset_path_complete, header=None)
    # add column names based on length of the dataset
    data.columns = [f'feature_{i+1}' for i in range(data.shape[1])]


    # Display the first 5 rows of the dataset
    print(data.head())
    # Display the shape of the dataset
    print(data.shape)
    # Display the column names
    print(data.columns)
    # Display the data types of the columns
    print(data.dtypes)
    # Display the number of missing values in each column
    print(data.isnull().sum())
    # Display the summary statistics of the dataset
    print(data.describe())

    return data.head(), data.shape, data.columns, data.dtypes, data.isnull().sum(), data.describe() , data


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
    return datafiles_path



def app():
    st.title('📊 Análise de Datasets')

    # Input do usuário para o link do dataset
    dataset_url = st.text_input('🔗 Insira o link do dataset UCI', 'https://archive.ics.uci.edu/dataset/53/iris')
    
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

        # Let the user select two variables from the dataset
        if selected_dataset is not None and not selected_dataset.empty:
            available_columns = selected_dataset.columns.tolist()
            var1 = st.selectbox('Select the first variable for comparison:', available_columns, index=0)
            var2 = st.selectbox('Select the second variable for comparison:', available_columns, index=min(1, len(available_columns)-1))
            
            # Assuming 'selected_file' is the file selected by the user
            if 'dataset_path_complete' in st.session_state :

                # Call the modified function with the selected dataset
                getTestDone(selected_dataset, var1, var2)

                # Display var information
                st.write('Informações sobre as variáveis selecionadas:')
                st.write(f'Variável 1: {var1}')
                st.write(f'Variável 2: {var2}')
                


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

   


if __name__ == "__main__":
    app()