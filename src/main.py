import numpy as np
import scipy.stats as stats

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 
  
# metadata 
print(predict_students_dropout_and_academic_success.metadata) 
  
# variable information 
print(predict_students_dropout_and_academic_success.variables) 


# Exemplo de teste qui-quadrado
def teste_qui_quadrado(observed_values, expected_values):
    chi_square_statistic, p_value = stats.chisquare(f_obs=observed_values, f_exp=expected_values)
    print("Teste Qui-Quadrado")
    print(f"Estatística de teste: {chi_square_statistic}")
    print(f"Valor-p: {p_value}")
    # Adicione sua lógica para interpretar o resultado baseado no nível de significância

# Exemplo de Teste t de Student para duas amostras independentes
def teste_t_student(amostra1, amostra2):
    stat, p_value = stats.ttest_ind(amostra1, amostra2)
    print("Teste t de Student para duas amostras independentes")
    print(f"Estatística de teste: {stat}")
    print(f"Valor-p: {p_value}")
    # Adicione sua lógica para interpretar o resultado baseado no nível de significância

# Exemplo de Teste ANOVA para múltiplas amostras
def teste_anova(*amostras):
    stat, p_value = stats.f_oneway(*amostras)
    print("Teste ANOVA")
    print(f"Estatística de teste: {stat}")
    print(f"Valor-p: {p_value}")
    # Adicione sua lógica para interpretar o resultado baseado no nível de significância

# Exemplos de chamadas das funções
# Você precisará substituir os valores das amostras pelos seus dados específicos
teste_qui_quadrado([10, 20, 30], [15, 15, 30])
teste_t_student(np.random.normal(loc=5, scale=2, size=100), np.random.normal(loc=5, scale=2.5, size=100))
teste_anova(np.random.normal(loc=5, scale=2, size=100), np.random.normal(loc=5, scale=2, size=100), np.random.normal(loc=5.5, scale=2, size=100))
