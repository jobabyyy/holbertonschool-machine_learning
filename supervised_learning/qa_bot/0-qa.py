#!/usr/bin/env python3
"""Q & A Bot: Question Answer"""


import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertForQuestionAnswering


def question_answer(input_question, input_reference):
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
    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = TFBertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Tokenize input
    question_tokens = tokenizer.tokenize(input_question)
    reference_tokens = tokenizer.tokenize(input_reference)

    # Prepare tokens for the model
    bert_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + reference_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
    token_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(reference_tokens) + 1)
    attention_mask = [1] * len(input_ids)

    # Convert inputs to tensors
    inputs = {
        'input_ids': tf.convert_to_tensor([input_ids], dtype=tf.int32),
        'token_type_ids': tf.convert_to_tensor([token_type_ids], dtype=tf.int32),
        'attention_mask': tf.convert_to_tensor([attention_mask], dtype=tf.int32),
    }

    # Get model outputs
    outputs = model(inputs)

    # Get start and end of the answer
    answer_start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    answer_end = tf.argmax(outputs.end_logits, axis=1).numpy()[0] + 1

    # Get answer tokens and convert them to a string
    answer_token = inputs['input_ids'].numpy()[0][answer_start: answer_end]
    answer = tokenizer.decode(answer_token)

    # Handle the case of no answer, return None
    if not answer.strip():
        return None

    return answer

if __name__ == "__main__":

    file_path = ('/content/drive/MyDrive/ZendeskArticles/PeerLearningDays.md')
    
    with open(file_path, 'r') as f:
      reference = f.read()

    print(question_answer('When are PLDs?', reference))
