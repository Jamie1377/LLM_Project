---
language: en
license: mit
tags:
- crypto
- sentiment-analysis
- inference-endpoint
- custom-handler
---

# Crypto News Custom Inference Endpoint

This repo is endpoint-ready for custom multi-input inference:
- text (news)
- btc_price_now
- fng_value
- fng_classification

Output fields:
- pred_class
- sentiment
- score
- prob_up
- confidence

## Deploy on Hugging Face Inference Endpoints

1. Go to Inference Endpoints and create a new endpoint from this model repo.
2. Task/engine: use custom handler mode (repository contains `handler.py`).
3. Ensure environment installs `requirements.txt`.
4. Deploy.

## API Payload Example

```json
{
  "inputs": {
    "text": "Bitcoin ETF inflows rose this week as institutional demand accelerated.",
    "btc_price_now": 67000,
    "fng_value": 62,
    "fng_classification": "Greed"
  }
}
```

## Python Snippet

```python
import requests

ENDPOINT_URL = "https://YOUR_ENDPOINT_URL"
TOKEN = "hf_xxx"

payload = {
    "inputs": {
        "text": "Bitcoin ETF inflows rose this week as institutional demand accelerated.",
        "btc_price_now": 67000,
        "fng_value": 62,
        "fng_classification": "Greed"
    }
}

resp = requests.post(
    ENDPOINT_URL,
    headers={"Authorization": f"Bearer {TOKEN}"},
    json=payload,
    timeout=60,
)
resp.raise_for_status()
print(resp.json())
```
