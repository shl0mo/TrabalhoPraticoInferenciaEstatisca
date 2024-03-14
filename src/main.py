import numpy as np
import pandas as pd
import scipy.stats as stats
from ucimlrepo import fetch_ucirepo

# Fetch dataset
adult = fetch_ucirepo(id=2)
data = adult.data.original

# Tratar valores ausentes
data = data.dropna()  # Ou substitua os valores ausentes por uma estratégia adequada

# Separar recursos e alvo
X = data.drop('income', axis=1)
y = data['income']

# Codificar variáveis categóricas em numéricas
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Função para verificar pressupostos
def check_assumptions(data, test):
    # Lógica para verificar pressupostos do teste
    if test == 'teste_t':
        # Verificar se as variâncias são iguais
        var1 = np.var(data[:len(data)//2])
        var2 = np.var(data[len(data)//2:])
        if var1/var2 > 2 or var2/var1 > 2:
            print("As variâncias são diferentes")
        else:
            print("As variâncias são iguais")
    elif test == 'teste_anova':
        # Verificar se as variâncias são iguais
        variances = [np.var(amostra) for amostra in data]
        if max(variances)/min(variances) > 2:
            print("As variâncias são diferentes")
        else:
            print("As variâncias são iguais")



# Exemplo de teste qui-quadrado
def teste_qui_quadrado(feature_name, alpha=0.05):
    feature = X[feature_name]
    observed_values, bins = np.histogram(feature, bins=10)
    expected_values = np.ones_like(observed_values) * len(feature) / 10
    chi_square_statistic, p_value = stats.chisquare(f_obs=observed_values, f_exp=expected_values)
    print(f"Teste Qui-Quadrado para a feature '{feature_name}'")
    print(f"Estatística de teste: {chi_square_statistic}")
    print(f"Valor-p: {p_value}")
    if p_value < alpha:
        print("Rejeita-se a hipótese nula")
    else:
        print("Não se rejeita a hipótese nula")

# Exemplo de Teste t de Student para duas amostras independentes
def teste_t_student(feature_name, alpha=0.05):
    feature = X[feature_name]
    amostra1 = feature[y == '>50K']
    amostra2 = feature[y == '<=50K']
    check_assumptions(pd.concat([amostra1, amostra2]), 'teste_t')
    stat, p_value = stats.ttest_ind(amostra1, amostra2)
    print(f"Teste t de Student para a feature '{feature_name}'")
    print(f"Estatística de teste: {stat}")
    print(f"Valor-p: {p_value}")
    if p_value < alpha:
        print("Rejeita-se a hipótese nula")
    else:
        print("Não se rejeita a hipótese nula")

# Exemplo de Teste ANOVA para múltiplas amostras
def teste_anova(feature_name, alpha=0.05):
    feature = X[feature_name]
    amostras = []
    for target_value in y.unique():
        amostras.append(feature[y == target_value])
    check_assumptions(feature, 'teste_anova')
    stat, p_value = stats.f_oneway(*amostras)
    print(f"Teste ANOVA para a feature '{feature_name}'")
    print(f"Estatística de teste: {stat}")
    print(f"Valor-p: {p_value}")
    if p_value < alpha:
        print("Rejeita-se a hipótese nula")
    else:
        print("Não se rejeita a hipótese nula")

# Exemplos de chamadas das funções
teste_qui_quadrado('age')
teste_t_student('age')
teste_anova('age')