#!/usr/bin/env python3
"""Q & A Bot: Answer Questions"""


from sentence_transformers import SentenceTransformer, util
import os
import tensorflow as tf
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """function that performs semanric search
    on a corpus of documents.

    Args:
        corpus_path: path to the corpus of reference
                     documents on which to perform
                     semantic search
        sentence: the sentence from which to perform
                      semantic search
    Returns: the reference text of the document most
             similar to sentence
    """
    # load the pre-trained model for generating sentence embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # load the reference documents
    corpus = []
    for filename in os.listdir(corpus_path):
        with open(os.path.join(corpus_path, filename), 'r', encoding='iso-8859-1') as file:
            text = file.read()
            corpus.append(text)

    if not corpus:
        raise ValueError("No files found in the corpus directory.")

    # compute embeddings
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    # Calc cosine similarities
    similarities = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)

    # retrieve the index of the most similar document
    most_similar_idx = similarities.argmax().item()

    # return the reference text of the most similar document and the filename
    most_similar_document = corpus[most_similar_idx]

    return most_similar_document

# update the corpus_path variable to point to the correct directory
corpus_path = (
    '/Users/jobabyyy/Desktop/ML T1/MachineLearningGH/holbertonschool-machine_learning/supervised_learning/qa_bot/ZendeskArticles')
query_sentence = ""
try:
    most_similar_document = semantic_search(corpus_path, query_sentence)

    print(most_similar_document)

except ValueError as e:
    print("Error:", e)
