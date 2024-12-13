import nltk
import re
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Preprocess documents and display cleaned documents on output         
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)


def process(doc1, doc2):
    cleaned_doc1 = preprocess(doc1)
    cleaned_doc2 = preprocess(doc2)

    print("Cleaned Doc 1: ", cleaned_doc1)
    print("Cleaned Doc 2: ", cleaned_doc2)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_doc1, cleaned_doc2])
    
    # Generate TF-IDF Vectors for each Doc and display on output 
    print("Feature Names: ", vectorizer.get_feature_names_out())
    print("Doc 1 Vector: ", tfidf_matrix[0].toarray())
    print("Doc 2 Vector: ", tfidf_matrix[1].toarray())
    
    # Compute cosine similarity and display on output
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    print("Cosine Similarity: ", similarity[0][0])
    
    # Discussion on similarity 
    if similarity[0][0] > 0.5:
        print("The documents are similar due to common themes in the text.")
    else:
        print("The documents are not similar due to different context and subjects in each document.")
    
    
def main():
    doc1 = """Mr Jeremy put on a macintosh, and a pair of shiny shoes; he took his fishing rod and basket, 
           and set off with enormous hops to the place where he kept his boat. The boat was round and green, 
           and very like the other lily-leaves. It was tied to a water-plant in the middle of the pond."""
    doc2 = """Peter never stopped running or looked behind him till he got home to the big fir-tree. 
           He was so tired that he flopped down upon the nice soft sand on the floor of the rabbit-hole 
           and shut his eyes. His mother was busy cooking; she wondered what he had done with his clothes. 
           It was the second little jacket and pair of shoes that Peter had lost in a week!"""
           
    process(doc1, doc2)
    
if __name__ == "__main__":
    main()


