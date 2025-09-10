from huggingface_hub import InferenceClient

HF_TOKEN=""#insert your api key here to check if the api is working. Select finegrained in hugging face under access tokens and select make calls to inference providers. and generate key and use it
MODEL = "deepseek-ai/DeepSeek-V3-0324"
client = InferenceClient(api_key=HF_TOKEN, provider="auto")

messages = [{"role": "user", "content": "Tell me a short joke."}]

resp = client.chat_completion(
    model=MODEL,
    messages=messages,
    max_tokens=100,
    temperature=0.7
)

print("Chat response:", resp)
