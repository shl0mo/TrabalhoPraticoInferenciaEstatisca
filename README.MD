# Análise de Datasets

Este é um aplicativo Streamlit para analisar datasets do repositório UCI Machine Learning Repository. O aplicativo permite baixar, visualizar, explorar e realizar diferentes análises estatísticas em datasets selecionados.

## Funcionalidades

- Baixar datasets diretamente do repositório UCI usando o link fornecido
- Visualizar informações gerais do dataset, como número de linhas, colunas, tipos de dados e estatísticas descritivas
- Explorar os dados por meio de histogramas e box plots para variáveis numéricas
- Realizar análises estatísticas, incluindo:
  - Teste Qui-Quadrado
  - Teste T de Student
  - Análise de Variância (ANOVA)
  - Análise de Correlação
  - Análise de Regressão Linear

## Requisitos

- Python 3.x
- Bibliotecas: pandas, numpy, scipy, requests, zipfile, matplotlib, streamlit, scikit-learn

## Uso

1. Clone o repositório ou faça o download do código-fonte.
2. Instale as bibliotecas necessárias: `pip install -r requirements.txt`
3. Execute o aplicativo Streamlit: `streamlit run app.py`
4. O aplicativo será aberto em seu navegador padrão.
5. Insira o link do dataset UCI que deseja analisar e siga as instruções.

## Contribuição

Contribuições são bem-vindas! Se você encontrar algum problema ou tiver sugestões de melhoria, sinta-se à vontade para abrir uma nova issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).