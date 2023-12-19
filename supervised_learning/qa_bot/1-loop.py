#!/usr/bin/env python3
"""Q & A Bot: Create the loop"""


while True:
    """
    Script to take input
    from the user with the prompt Q:
    and prints A: as a response.

    If the user inputs: exit, quit, goodbye, or bye
    case, insensitive, print A: Goodbye and exit.
    """
    user_input = input("Q: ").strip().lower()

    if user_input in ["exit", "quit", "goodbye", "bye"]:
        print("A: Goodbye")
        break

    print("A:")
