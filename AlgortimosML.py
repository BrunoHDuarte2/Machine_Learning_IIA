import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
file_path = 'LLM.csv'
df = pd.read_csv(file_path)
df = df.dropna(subset=['Label'])
texts = df['Text'].tolist()
labels = df['Label'].tolist()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
# Matriz TF-IDF = Frequency Inverse Document -> Frequencia de cada palavra com o intuito de indicar sua importância
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
# Divisão do conjunto de dados (30% teste, 70% treino)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
###################################### Aplicação do LDA ####################################################
lda = LinearDiscriminantAnalysis()
lda.fit(X_train.toarray(), y_train)  

# Predições para métricas
y_pred = lda.predict(X_test.toarray())
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.3f}')

y_score = lda.decision_function(X_test.toarray())

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - LDA')
plt.legend(loc="lower right")
plt.savefig('roc_curve_lda.png')


###################################### Aplicação do QDA ####################################################
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train.toarray(), y_train)
# Obter as previsões
y_pred = qda.predict(X_test.toarray())

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.3f}')

# Obter as previsões de probabilidade
y_score = qda.decision_function(X_test.toarray())

# Calcular a curva ROC e AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - QDA')
plt.legend(loc="lower right")

# Salvar a figura como imagem
plt.savefig('roc_curve_qda.png')

###################################### Aplicação do KNN ####################################################
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train.toarray(), y_train) 
y_pred = knn.predict(X_test.toarray())

accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.3f}')

y_score = knn.predict_proba(X_test.toarray())[:, 1]  

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - KNN')
plt.legend(loc="lower right")

# Salvar a figura como imagem
plt.savefig('roc_curve_knn.png')

