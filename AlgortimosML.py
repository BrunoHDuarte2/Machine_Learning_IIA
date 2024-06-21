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
#####
from sklearn.metrics import roc_curve, auc, accuracy_score
# Leitura e separação dos dados e rótulos
file_path = 'LLM.csv'
df = pd.read_csv(file_path)
df = df.dropna(subset=['Label'])
texts = df['Text'].tolist()
labels = df['Label'].tolist()
# Transformação dos rótulos em 0 ou 1 -> Somente duas classes (AI->0 ou Student->1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
# Matriz TF-IDF = Frequency Inverse Document -> Frequencia de cada palavra com o intuito de indicar sua importância
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
# Divisão do conjunto de dados (30% teste, 70% treino)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3)

###################################### Plot dos dados usando PCA ###########################################
# PCA é usado para reduzir as dimensões dos dados para ser possível plota-los
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded)
plt.title('Dados')
plt.colorbar(scatter, label='AI                                                                Student')
plt.savefig('dados.png')
###################################### Aplicação do LDA ####################################################
lda = LinearDiscriminantAnalysis()
# Aplicação do modelo com os dados de treino
lda.fit(X_train.toarray(), y_train)  
# Predições para métricas com os dados de teste
y_pred = lda.predict(X_test.toarray())
# Precisão do modelo 
acc_lda = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo de Linear Discriminant Analysis(LDA) sobre os dados: {acc_lda:.10f}')
# Probabilidades previstas -> Usado no roc_curve para obter a taxa de falsos positivos e verdadeiros positivos
y_score = lda.decision_function(X_test.toarray())
# Obtençaõ das taxas necessária para se plotar a roc curve
fpr, tpr, threshold = roc_curve(y_test, y_score)
# Area Under Curve calculada com as taxas 
roc_auc_lda = auc(fpr, tpr)
# Plot da Roc curve, a imagem é gerada e salva no projeto quando executado!
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.15f)' % roc_auc_lda)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - LDA')
plt.legend(loc="lower right")
plt.savefig('roc_curve_lda.png')

###################################### Aplicação do QDA ####################################################
qda = QuadraticDiscriminantAnalysis()
# Aplicação do QDA aos dados de treino
qda.fit(X_train.toarray(), y_train)
# Obter as previsões
y_pred = qda.predict(X_test.toarray())
# Calcular a precisão 
acc_qda = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo de Quadratic Discriminant Analysis (QDA) sobre os dados: {acc_qda:.10f}')
# Obter as previsões de probabilidade
y_score = qda.decision_function(X_test.toarray())
# Calcular a curva ROC e AUC
fpr, tpr, threshold = roc_curve(y_test, y_score)
roc_auc_qda = auc(fpr, tpr)
# Plotar a curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.15f)' % roc_auc_qda)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - QDA')
plt.legend(loc="lower right")
plt.savefig('roc_curve_qda.png')

###################################### Aplicação do KNN ####################################################
# Foi utilizado a quantidade de vizinhos como 21 pois foi a que apresentou melhores precisões dentre todos os testes feitos
knn = KNeighborsClassifier(n_neighbors=21)
# Aplicação do KNN sobre os dados de treino
knn.fit(X_train.toarray(), y_train) 
# Predições com os dados de treino
y_pred = knn.predict(X_test.toarray())
# Calculo da precisão 
acc_knn = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo de K nearest neighbors (KNN) sobre os dados: {acc_knn:.10f}')
# Probabilidades da primeira class -> por isso o [:,1]
y_score = knn.predict_proba(X_test.toarray())[:, 1]
# Taxa de verdadeiros positivos e falsos positivos
fpr, tpr, threshold = roc_curve(y_test, y_score)
roc_auc_knn = auc(fpr, tpr)
# Plot da curva Roc
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.15f)' % roc_auc_knn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - KNN')
plt.legend(loc="lower right")
plt.savefig('roc_curve_knn.png')

###################################### Aplicação do SVM ####################################################
# Instancia de SVM -> probability = True para possibilitar o calculo das probabilidades para curva roc
svm = SVC(kernel='linear', probability=True)
# Aplicação de SVM sobre os dados de treino
svm.fit(X_train.toarray(), y_train)
# Predições com os dados de teste
y_pred = svm.predict(X_test.toarray())
# Calculo da precisão 
acc_svm = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo de Support Vector Machines (SVM) sobre os dados: {acc_svm:.10f}')
# Probabilidades da primeira class -> por isso o [:,1]
y_score = svm.predict_proba(X_test.toarray())[:, 1]
fpr, tpr, threshold = roc_curve(y_test, y_score)
roc_auc_svm = auc(fpr, tpr)
# Plot da curva Roc
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.15f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - SVM')
plt.legend(loc="lower right")
plt.savefig('roc_curve_svm.png')

###################################### Aplicação do Random Forest ####################################################
# Nesse caso, n_estimators especifica a quantidade de arvores de decisão que deverão ser feitas
# Após alguns testes, percebi que mesmo aumentando o valor de arvores a precisão não aumentou! 
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train.toarray(), y_train)
# Predições 
y_pred = rfc.predict(X_test.toarray())
# Precisão do modelo
acc_rfc = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo de Random Forest sobre os dados: {acc_rfc:.10f}')
# Probabilidades da primeira class -> por isso o [:,1]
y_score = rfc.predict_proba(X_test.toarray())[:, 1]
# Taxa de falsos positivos e verdadeiros positivos
fpr, tpr, threshold = roc_curve(y_test, y_score)
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
# Está sendo usado a AUC calculada para cada ROC!
roc_aucs= [(roc_auc_svm, "Support Vector Machine"), (roc_auc_knn, "K Nearest Neighbors"), (roc_auc_lda, "Linear Discriminant Analysis"), (roc_auc_qda, "Quadratic Discriminant Analysis"), (roc_auc_rfc, "Random Forest")]
roc_aucs_sort = sorted(roc_aucs, key=lambda x: x[0])
roc_aucs_sort = roc_aucs_sort[::-1]
for i in range(len(roc_aucs_sort)):
    acc, model = roc_aucs_sort[i]
    print(f'{i+1}°: {model}, com precisão de: {acc}')
