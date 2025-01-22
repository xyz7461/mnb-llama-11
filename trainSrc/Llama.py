from together import Together
import os
import requests
import json

def llama32(messages, model_size=11):
  model = f"meta-llama/Llama-3.2-{model_size}B-Vision-Instruct-Turbo"
  url = "https://api.together.xyz/v1/chat/completions"
  payload = {
    "model": model,
    "max_tokens": 4096,
    "temperature": 0.0,
    "stop": ["<|eot_id|>","<|eom_id|>"],
    "messages": messages
  }

  headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer " + os.environ["TOGETHER_API_KEY"]
  }
  res = json.loads(requests.request("POST", url, headers=headers, data=json.dumps(payload)).content)

  if 'error' in res:
    raise Exception(res['error'])

  return res['choices'][0]['message']['content']

messages = [
  {
    "role": "user",
    "content": "what are large language models?"
  }
]

response = llama32(messages)
print(response)
