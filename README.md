# Detecção de Texto Gerado por IA vs Texto Gerado por Estudante
## Descrição
Este projeto utiliza algoritmos de aprendizado de máquina para classificar textos como gerados por Inteligência Artificial (IA) ou por estudantes. Os algoritmos aplicados incluem LDA, QDA, KNN, Random Forest e SVM. Os dados utilizados foram obtidos do [Kaggle](https://www.kaggle.com/datasets/prajwaldongre/llm-detect-ai-generated-vs-student-generated-text). Para a extração de características textuais, foi utilizada a técnica TF-IDF (Term Frequency-Inverse Document Frequency).

## Estrutura do Repositório
- AlgoritmosML.py contém a extração e aplicação dos modelos de Machine Learning. Ao executá-lo é gerado 6 imagens: 
    - dados.png: imagem gerada usando PCA e são os dados textuais plotados de forma a conseguir visualizar como cada algoritmo separaria esses dados.
    - roc_curve_knn.png: imagem gerada da curva ROC do algoritmo KNN.
    - roc_curve_lda.png: imagem gerada da curva ROC do algoritmo LDA.
    - roc_curve_qda.png: imagem gerada da curva ROC do algoritmo QDA.
    - roc_curve_rfc.png: imagem gerada da curva ROC do algoritmo Random Forest.
    - roc_curve_svm.png: imagem gerada da curva ROC do algoritmo SVM.
- Teste_Knn.py contém um algoritmo para ajudar na decisão de qual k usar no algoritmo KNN, isto é, o número de vizinhos a ser usados.
- Teste_Knn.py contém um algoritmo para ajudar na decisão de qual n_estimators usar no algoritmo Random Forest, isto é, o número de Arvores de Decisão a ser usados.
- Texto_Explicativo.txt: contém uma explicação breve do código e ranking com quais algoritmos se sairam melhor para esse conjunto de dados.
- LLM.csv: contém os dados brutos, obtidos no Kaggle.

## Pré-requisitos 
Para rodar o código é preciso de algumas bibliotecas do python, são elas:
- scikit-learn
- matplotlib
- pandas
Caso não tenha instalado, você pode instalar executando:
```bash
pip install pandas scikit-learn matplotlib
```