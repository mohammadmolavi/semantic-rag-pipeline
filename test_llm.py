from dotenv import load_dotenv
import os
import requests

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY Not Found!")


url = "https://openrouter.ai/api/v1/chat/completions"


headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer in Persian.",
        },
        {
            "role": "user",
            "content": "در سه جمله توضیح بده مدل زبانی بزرگ چیست.",
        },
    ],
    "temperature": 0.7,
    "max_tokens": 300,
}

try:
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()
    result = response.json()

    print("Selected Model is:", result.get("model", "Unknown"))
    print("\nAnswer:")
    print(result["choices"][0]["message"]["content"])

except requests.HTTPError:
    print("API Error:", response.status_code)
    print(response.text)

except requests.RequestException as error:
    print("Communication Error:", error)
