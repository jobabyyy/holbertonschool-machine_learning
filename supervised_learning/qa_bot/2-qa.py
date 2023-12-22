#!/usr/bin/env python3
"""Q & A Bot: Answer Questions"""


import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer, TFAutoModelForQuestionAnswering


def question_answer(question, reference):
    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = TFAutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Tokenize the question and reference
    inputs = tokenizer(question, reference, return_tensors="tf", padding=True, truncation=True)

    outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Find the answer span
    start_index = tf.argmax(start_scores, axis=1).numpy()[0]
    end_index = tf.argmax(end_scores, axis=1).numpy()[0] + 1

    # Convert answer tokens to a human-readable string
    answer_tokens = inputs["input_ids"].numpy()[0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens)

    # Case handling if there is no answer
    if not answer.strip():
        return None

    return answer

def answer_loop(reference):
    """Summary: answers questions from reference text."""
    while True:
        user_input = input("Q: ").strip().lower()

        if user_input in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        answer = question_answer(user_input, reference)

        if answer is not None:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")

if __name__ == "__main__":
    file_path = ('/content/drive/MyDrive/ZendeskArticles/PeerLearningDays.md')

    with open(file_path, 'r') as f:
        reference = f.read()

    answer_loop(reference)
