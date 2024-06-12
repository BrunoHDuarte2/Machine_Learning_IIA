import pandas as pd
file_path = 'LLM.csv'
df = pd.read_csv(file_path)
texts = []
authors = []
for index, row in df.iterrows():
    texto = row['Text']
    label = row['Label']
    texts.append(texto)
    authors.append(label)

dados = zip(texts, authors)
for x, y in dados:
    print(f"Texto: {x}, Label: {y}")