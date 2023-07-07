#!/bin/bash

URL=https://replit-code-v1-inference-service-predictor-default.tenant-ae3e9a-cw1.knative.chi.coreweave.com
KEY=your_api_key_here
ENDPOINT="$URL/v1/models/replit-code-v1:predict"

echo $ENDPOINT
curl -v -X POST "$ENDPOINT" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $KEY" \
     -d '{"instances": [{"prompt": "def fibonacci(n):", "max_tokens": 512, "n": 1, "temperature": 0.2, "stop": "<|endoftext|>"}]}'
