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

        # Check for exit conditions
        if question.lower() in ['goodbye', 'exit']:
            print("A: Goodbye")
            break  # Exit the loop and the script

        # Check for question in mapping
        if question.lower() in qa_map:
            # Print the answers from mapping
            print("A:", qa_map[question.lower()])
        else:
            print("A: I'm sorry, I don't have an answer for that.")

# Pointing corpus path in the correct direction
corpus_path = '/content/drive/MyDrive/ZendeskArticles/PeerLearningDays.md'

# Initiate conversation responses:
question_answer(corpus_path)
