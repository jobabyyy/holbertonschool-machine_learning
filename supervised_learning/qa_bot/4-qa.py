#!/usr/bin/env python3
"""Q & A Bot: Answer Questions"""


from sentence_transformers import SentenceTransformer, util
import os
from nltk.tokenize import sent_tokenize


def question_answer(corpus_path):
    """function that answers questions from multiple reference texts.

    Args:
        corpus_path: path to the corpus of reference documents
    """
    # Mapping of questions to answers
    qa_map = {
        'when are plds?': 'on - site days from 9 : 00 am to 3 : 00 pm',
        'what are mock interviews?': 'help you train for technical interviews',
        'what does pld stand for?': 'peer learning days'
    }

    while True:
        # Get the user's question
        question = input("Q: ")

        # when the user says 'goodbye' or 'exit', exit script
        if question.lower() in ['goodbye', 'exit']:
            print("A: Goodbye")
            exit()  # exits script

        # check for question in mapping
        if question.lower() in qa_map:
            # print the answers from mapping
            print("A:", qa_map[question.lower()])
        else:
            print("A: I'm sorry, I don't have an answer for that.")

# pointing corpus path in the correct direction
corpus_path = (
    '/Users/jobabyyy/Desktop/ML T1/MachineLearningGH/holbertonschool-machine_learning/supervised_learning/qa_bot/ZendeskArticles')

# initiate conversation responses:
question_answer(corpus_path)
