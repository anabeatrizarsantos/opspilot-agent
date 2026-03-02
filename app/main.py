from llm_client import ask_llm

def main():
    """Simple CLI loop to interact with OpsPilot."""

    print("OpsPilot CLI started. Type 'exit' to stop.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = ask_llm(user_input)
        print("OpsPilot:", response)


if __name__ == "__main__":
    main()