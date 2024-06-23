import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

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

#Possíveis números de florestas
F = [x for x in range(1, 200, 25)]
iteracoes = 200


melhor_auc = [0, 0]
pior_auc = [1, 0]
for f in F:

    auc_media = 0

    for i in range(iteracoes):
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3)

        lda = RandomForestClassifier(n_estimators=f)
        lda.fit(X_train.toarray(), y_train)

        y_score = lda.predict_proba(X_test.toarray())[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, y_score)
        roc_auc_lda = auc(fpr, tpr)

        auc_media += roc_auc_lda

    auc_media /= iteracoes
    print(f"AUC: {auc_media:.5f} (florestas = {f})")

    if melhor_auc[0] < auc_media:
        melhor_auc[0] = auc_media
        melhor_auc[1] = f

    if pior_auc[0] > auc_media:
        pior_auc[0] = auc_media
        pior_auc[1] = f


print(f"O melhor valor encontrado foi de\nF= {melhor_auc[1]}, com AUC de {melhor_auc[0]}")
print(f"O pior valor encontrado foi de\nF= {pior_auc[1]}, com AUC de {pior_auc[0]}")