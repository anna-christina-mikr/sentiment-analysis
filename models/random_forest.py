def return_to_tensor(txt):
    return eval("torch." + txt).numpy()


imdb_df = pd.read_csv('../imdb_with_glove_bert_embeddings.csv')
imdb_df['cls_bert'] = imdb_df['cls_bert'].apply(return_to_tensor)
imdb_df.drop('encode_glove', axis=1, inplace=True)

X = imdb_df['review']
y = imdb_df['sentiment']
X_bert = np.vstack(imdb_df['cls_bert'])
X_glove = get_glove_embedding(imdb_df)
labs = [1 if label == "positive" else 0 for label in y]
labels = torch.tensor(labs).float().unsqueeze(1)


imdb_df['sentiment'] = imdb_df['sentiment'].replace({'negative': 0, 'positive': 1})
X = imdb_df['review'].tolist()
y = imdb_df['sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
def get_bert_embeddings(texts, tokenizer, model, max_length=128):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.numpy()
X_train_embeddings = get_bert_embeddings(X_train, tokenizer, bert_model)
X_test_embeddings = get_bert_embeddings(X_test, tokenizer, bert_model)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_embeddings, y_train)

y_pred = rf_model.predict(X_test_embeddings)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



#GloVe
def load_glove_embeddings(GloveFile):

    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove_path = "/path/to/glove.6B.50d.txt" 
glove_embeddings = load_glove_embeddings(GloveFile)
print(f"Loaded {len(glove_embeddings)} word vectors.")




def get_glove_embeddings(texts, glove_embeddings, embedding_dim=50):

    embeddings = []
    for text in texts:
        words = text.split()
        word_vectors = [glove_embeddings[word] for word in words if word in glove_embeddings]
        if word_vectors:
            embeddings.append(np.mean(word_vectors, axis=0))
        else:
            embeddings.append(np.zeros(embedding_dim))
    return np.array(embeddings)


embedding_dim = 50 
X_train_embeddings = get_glove_embeddings(X_train, glove_embeddings, embedding_dim=embedding_dim)
X_test_embeddings = get_glove_embeddings(X_test, glove_embeddings, embedding_dim=embedding_dim)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_embeddings, y_train)

y_pred = rf_model.predict(X_test_embeddings)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


#BOW
vectorizer = CountVectorizer(max_features=5000)  # Limit features to 5000 for efficiency
X_train_bow = vectorizer.fit_transform(X_train).toarray()
X_test_bow = vectorizer.transform(X_test).toarray()

rf_model_bow = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_bow.fit(X_train_bow, y_train)

y_pred_bow = rf_model_bow.predict(X_test_bow)
print("Accuracy:", accuracy_score(y_test, y_pred_bow))
print("Classification Report:\n", classification_report(y_test, y_pred_bow))


#TF-IDF 
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to 5000 for efficiency
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

rf_model_tfidf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_tfidf.fit(X_train_tfidf, y_train)

y_pred_tfidf = rf_model_tfidf.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print("Classification Report:\n", classification_report(y_test, y_pred_tfidf))
