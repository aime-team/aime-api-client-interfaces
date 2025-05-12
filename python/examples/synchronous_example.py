import json
from aime_api_client_interface import ModelAPI

def main():
    model_api = ModelAPI('https://api.aime.info', 'llama3_chat', 'apiexample@aime.info', '181e35ac-7b7d-4bfe-9f12-153757ec3952')
    model_api.do_api_login()

    chat_context = [
        {"role": "user", "content": "Hi! How are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"}
    ]

    params = {
        "prompt_input": "Tell me a joke",
        "chat_context": json.dumps(chat_context),
        "top_k": 40,
        "top_p": 0.9,
        "temperature": 0.8,
        "max_gen_tokens": 1000
    }

    result = model_api.do_api_request(params)
    print("Synchronous result:", result)

if __name__ == "__main__":
    main()