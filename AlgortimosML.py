import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
# Import dos modelos
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc, accuracy_score

# Leitura dos dados
file_path = 'LLM.csv'
df = pd.read_csv(file_path)

# Organização dos dados: retirada dos textos sem label, separação dos vetores de dados e classes
df = df.dropna(subset=['Label'])
texts = df['Text'].tolist()
labels = df['Label'].tolist()

# Transformação dos rótulos em valores numéricos -> "AI" = 0, "Student" = 1
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# Matriz TF-IDF
# TF-IDF é uma técnica popular de representar conjuntos de documentos de textos de forma numérica, em uma matriz.
# TF (Term Frequency) indica a frequência de uma palavra em um documento,
# e IDF (Inverse Document Frequency) indica a raridade de uma palavra considerando todo o conjunto de documentos.
# Dessa forma, é possivel estimar a medida de importância das palavras em cada texto.
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)     # Matriz TF-IDF

# Divisão (aleatória) do conjunto de dados -> 30% teste, 70% treino
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3)

# Plot dos dados
# PCA é usado para reduzir as dimensões dos dados, possibilitando plota-los
# Este gráfico foi incluído para visualização dos dados
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded)
plt.title('Dados')
plt.colorbar(scatter, label='AI                                                                Student')
plt.savefig('dados.png')

###################################### Aplicação do LDA ####################################################
# Cria uma instância do modelo de aprendizado LDA, e treina-o com os dados previamente separados
lda = LinearDiscriminantAnalysis()
lda.fit(X_train.toarray(), y_train)

# Aplica o modelo treinado nos dados de teste, gerando as predições de classes
y_pred = lda.predict(X_test.toarray())

# Acurácia do modelo
acc_lda = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo de Linear Discriminant Analysis(LDA) sobre os dados: {acc_lda:.10f}')

# Probabilidades previstas -> Usado no roc_curve para obter a taxa de falsos positivos e verdadeiros positivos
y_score = lda.decision_function(X_test.toarray())
# Obtenção das taxas de falsos-positivos e verdadeiros-positivos para se plotar a curva ROC
fpr, tpr, threshold = roc_curve(y_test, y_score)    # False-Positive Rate, True-Positive Rate
# Area Under Curve (AUC) calculada com as taxas
roc_auc_lda = auc(fpr, tpr)

# Plot gráfico da curva ROC
# O grafico é gerado, salvo no projeto quando executado
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.15f)' % roc_auc_lda)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - LDA')
plt.legend(loc="lower right")
plt.savefig('roc_curve_lda.png')

###################################### Aplicação do QDA ####################################################
# Cria uma instância do modelo de aprendizado QDA, e treina-o com os dados previamente separados
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train.toarray(), y_train)

# Aplição do modelo, gerando as predições de classes
y_pred = qda.predict(X_test.toarray())

# Acurácia do modelo
acc_qda = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo de Quadratic Discriminant Analysis (QDA) sobre os dados: {acc_qda:.10f}')

# Obter as previsões de probabilidade
y_score = qda.decision_function(X_test.toarray())
# Calcular a curva ROC e sua AUC
fpr, tpr, threshold = roc_curve(y_test, y_score)    # False-Positive Rate, True-Positive Rate
roc_auc_qda = auc(fpr, tpr)

# Plotar a curva ROC
# Novamente, o gráfico gerado é salvo no projeto como uma imagem
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.15f)' % roc_auc_qda)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - QDA')
plt.legend(loc="lower right")
plt.savefig('roc_curve_qda.png')

###################################### Aplicação do K-nn ####################################################
# A quantidade de vizinhos (K) foi escolhida como 13, pois foi a que apresentou melhores precisões dentre os testes realizados
# Cria uma instância do modelo de aprendizado K-nn, e treina-o com os dados previamente separados
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train.toarray(), y_train)

# Aplição do modelo, gerando as predições de classes
y_pred = knn.predict(X_test.toarray())

# Acurácia do modelo
acc_knn = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo de K nearest neighbors (KNN) sobre os dados: {acc_knn:.10f}')

# Probabilidades da classe 1
# A função predict_proba() gera as probabilidades de cada elemento pertencer a ambas as classes,
# então a matriz é cortada ([:,1]) para que se possa calcular a curva ROC
y_score = knn.predict_proba(X_test.toarray())[:, 1]
# Calcular a curva ROC e sua AUC
fpr, tpr, threshold = roc_curve(y_test, y_score)    # False-Positive Rate, True-Positive Rate
roc_auc_knn = auc(fpr, tpr)

# Plot da curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.15f)' % roc_auc_knn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - KNN')
plt.legend(loc="lower right")
plt.savefig('roc_curve_knn.png')

###################################### Aplicação do SVM ####################################################
# Cria uma instância do modelo de aprendizado SVM, e treina-o com os dados previamente separados
# O parâmetro 'probability=True' permite o cálculo das probabilidades (linha 156)
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train.toarray(), y_train)

# Aplição do modelo, gerando as predições de classes
y_pred = svm.predict(X_test.toarray())

# Acurácia do modelo
acc_svm = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo de Support Vector Machines (SVM) sobre os dados: {acc_svm:.10f}')

# Probabilidades da classe 1
# Novamente foi necessário separar as probabilidades calculadas para ambas as classes
y_score = svm.predict_proba(X_test.toarray())[:, 1]
# Calcular a curva ROC e sua AUC
fpr, tpr, threshold = roc_curve(y_test, y_score)    # False-Positive Rate, True-Positive Rate
roc_auc_svm = auc(fpr, tpr)

# Plot da curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.15f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - SVM')
plt.legend(loc="lower right")
plt.savefig('roc_curve_svm.png')

###################################### Aplicação do Random Forest ####################################################
# Cria uma instância do modelo de aprendizado Random Forest, e treina-o com os dados previamente separados
# O parâmetro "n_estimators=100" indica que deverão ser geradas 100 árvores do algorítimo Random Forest
# Nos testes realizados, um número de árvores maior que 100 não resultava em melhora de performance, por isso n foi escolhido como 100
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train.toarray(), y_train)

# Aplição do modelo, gerando as predições de classes
y_pred = rfc.predict(X_test.toarray())

# Acurácia do modelo
acc_rfc = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo de Random Forest sobre os dados: {acc_rfc:.10f}')

# Probabilidades da classe 1
# Novamente foi necessário separar as probabilidades calculadas para ambas as classes
y_score = rfc.predict_proba(X_test.toarray())[:, 1]
# Calcular a curva ROC e sua AUC
fpr, tpr, threshold = roc_curve(y_test, y_score)    # False-Positive Rate, True-Positive Rate
roc_auc_rfc = auc(fpr, tpr)

# Plot da curva Roc
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.15f)' % roc_auc_rfc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Random Forest')
plt.legend(loc="lower right")
plt.savefig('roc_curve_rfc.png')

####################################### Ordem de precisão dos algoritmos ###########################################
# Classificação dos melhores classificadores, para o conjunto de treino gerado
roc_aucs= [(roc_auc_svm, "Support Vector Machine"), (roc_auc_knn, "K Nearest Neighbors"), (roc_auc_lda, "Linear Discriminant Analysis"), (roc_auc_qda, "Quadratic Discriminant Analysis"), (roc_auc_rfc, "Random Forest")]
roc_aucs_sort = sorted(roc_aucs, key=lambda x: x[0])
roc_aucs_sort = roc_aucs_sort[::-1]

print("\n -----| Resultados |-----")
for i in range(len(roc_aucs_sort)):
    acc, model = roc_aucs_sort[i]
    print(f'{i+1}°: {model}, com AUC de: {acc}')
