#!/usr/bin/env python3
"""Q & A Bot: Question_Answer"""

import logging
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFAutoModelForQuestionAnswering

logging.getLogger('transformers').setLevel(logging.ERROR)

def question_answer(question, reference):
    """ function that finds a snippet of text
    within a reference doc to answer a question.

    Args:
        question (str): contains question to answer
        reference (str): containing the reference
                         document from which to find
                         the answer
    Returns: a string containing the answer 
    or NONE if no answer is found.
    """
    # load pre-train model and tokenizer
    model_name = (
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)

    # define function to perform question answering
    input = tokenizer(question, reference, padding=True, return_tensors="tf")
    output = model(**input)

    # convert vars to numpy arrays
    start_scores = output.start_logits
    end_scores = output.end_logits

    # get answer span
    start_answer = tf.argmax(start_scores, axis=1).numpy()[0]
    end_answer = tf.argmax(end_scores, axis=1).numpy()[0] + 1

    # convert predicted answer tokens to human readable string
    answer_token = input["input_ids"].numpy()[0][start_answer:end_answer]
    answer = tokenizer.decode(answer_token)

    # handle case of no answer, return None
    if not answer.strip():
        return None

    return answer
