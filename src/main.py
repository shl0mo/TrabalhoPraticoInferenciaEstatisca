import numpy as np
import pandas as pd
import scipy.stats as stats
from ucimlrepo import fetch_ucirepo

# Fetch dataset
adult = fetch_ucirepo(id=2)
data = adult.data.original

# Tratar valores ausentes
data = data.dropna()

# Separar recursos e alvo
X = data.drop('income', axis=1)
y = data['income']

# Manter a coluna 'workclass' original
workclass_col = X['workclass']

# Codificar variáveis categóricas em numéricas
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Adicionar a coluna 'workclass' de volta ao DataFrame X
X = pd.concat([X, workclass_col], axis=1)

# Teste 1: Qui-quadrado
# Descrição do problema: Verificar se a distribuição da variável 'age' segue uma distribuição uniforme.
# Hipóteses:
#   H0: A distribuição da variável 'age' segue uma distribuição uniforme.
#   H1: A distribuição da variável 'age' não segue uma distribuição uniforme.
# Nível de significância: 0.05
alpha = 0.05
feature_name = 'age'
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
print("\n")

# Teste 2: Teste t de Student para duas amostras independentes
# Descrição do problema: Verificar se há diferença significativa na média da variável 'age' entre as pessoas com renda <= 50K e > 50K.
# Hipóteses:
#   H0: Não há diferença significativa na média da variável 'age' entre as duas populações.
#   H1: Há diferença significativa na média da variável 'age' entre as duas populações.
# Nível de significância: 0.05
alpha = 0.05
feature_name = 'age'
feature = X[feature_name]
amostra1 = feature[y == '>50K']
amostra2 = feature[y == '<=50K']
stat, p_value = stats.ttest_ind(amostra1, amostra2)
print(f"Teste t de Student para a feature '{feature_name}'")
print(f"Estatística de teste: {stat}")
print(f"Valor-p: {p_value}")
if p_value < alpha:
    print("Rejeita-se a hipótese nula")
else:
    print("Não se rejeita a hipótese nula")
print("\n")

# Teste 3: Teste ANOVA para múltiplas amostras
# Descrição do problema: Verificar se há diferença significativa na média da variável 'education-num' entre as diferentes categorias da variável 'workclass'.
# Hipóteses:
#   H0: Não há diferença significativa na média da variável 'education-num' entre as categorias da variável 'workclass'.
#   H1: Há diferença significativa na média da variável 'education-num' entre pelo menos duas categorias da variável 'workclass'.
# Nível de significância: 0.05
alpha = 0.05
feature_name = 'education-num'
workclass_categories = X['workclass'].unique()
amostras = []
for category in workclass_categories:
    amostras.append(X.loc[X['workclass'] == category, feature_name])
stat, p_value = stats.f_oneway(*amostras)
print(f"Teste ANOVA para a feature '{feature_name}' e categorias da variável 'workclass'")
print(f"Estatística de teste: {stat}")
print(f"Valor-p: {p_value}")
if p_value < alpha:
    print("Rejeita-se a hipótese nula")
else:
    print("Não se rejeita a hipótese nula")