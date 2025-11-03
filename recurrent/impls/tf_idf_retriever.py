from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TfidfRetriever:
    """
    A class to handle TF-IDF retrieval using a Hugging Face tokenizer.
    The vectorizer is fitted once upon initialization.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vectorizer = TfidfVectorizer(tokenizer=self._llm_tokenizer)

    def _llm_tokenizer(self, text):
        """
        Custom tokenizer method that uses the instance's tokenizer.
        This method is passed to TfidfVectorizer.
        """
        lower_text = text.lower()
        
        # This is the line you asked about: using self.tokenizer
        tokens = self.tokenizer.tokenize(lower_text)
        
        # Normalize tokens to handle subword artifacts (like 'Ġ')
        normalized_tokens = [token.replace('Ġ', '') for token in tokens]
        return normalized_tokens

    def retrieve(self, query, corpus, top_k=3):
        """
        Retrieves the top_k most similar documents for a given query.
        """
        if not query or not corpus:
            return [(None, 0.0) for _ in range(top_k)]
        if not isinstance(corpus, list):
            corpus = list(corpus)
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except Exception as e:
            return [(None, 0.0) for _ in range(top_k)]
        
        q_vec = self.vectorizer.transform([query])

        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
        
        top_ids = np.argsort(sims)[::-1][:top_k]
        return [(corpus[i], sims[i]) for i in top_ids]
    
    def top1_retrieve(self, query, corpus):
        return self.retrieve(query, corpus, top_k=1)[0][0]
