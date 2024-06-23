import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve, auc

file_path = 'LLM.csv'
df = pd.read_csv(file_path)

df = df.dropna(subset=['Label'])
texts = df['Text'].tolist()
labels = df['Label'].tolist()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

#Possiveis valores de K, escolhidos a mão
K = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 33, 37]
iteracoes = 100

melhor_auc = [0, 0]
pior_auc = [1, 0]
for k in K:

    auc_media = 0

    for i in range(iteracoes):
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3)
        print(i)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train.toarray(), y_train)

        y_score = knn.predict_proba(X_test.toarray())[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, y_score)
        roc_auc_knn = auc(fpr, tpr)

        auc_media += roc_auc_knn

    auc_media /= iteracoes

    if melhor_auc[0] < auc_media:
        melhor_auc[0] = auc_media
        melhor_auc[1] = k

    if pior_auc[0] > auc_media:
        pior_auc[0] = auc_media
        pior_auc[1] = k


print(f"Dentre {K},\no melhor valor encontrado foi de\nK= {melhor_auc[1]}, com AUC de {melhor_auc[0]}")
print(f"O pior valor encontrado foi de\nK= {pior_auc[1]}, com AUC de {pior_auc[0]}")
