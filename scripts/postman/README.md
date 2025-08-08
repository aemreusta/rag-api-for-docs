# Postman usage

1. Import

- In Postman, click Import and add both files:
  - `Chatbot API Service.postman_collection.json`
  - `Chatbot API Service.postman_environment.json`

1. Select environment

- Choose `Chatbot API Service - Local` from the environment selector.

1. Variables

- `base_url`: defaults to `http://localhost:18000`.
- `api_key`: optional X-API-Key if you enabled auth.
- `provider`: one of `google`, `openrouter`, `groq`.
- `override_api_key`: to test a specific key with the validation endpoint.
- `session_id`: auto-generated if left empty.
- `question`: prompt text for the chat endpoint.

1. Requests

- Health: `GET {{base_url}}/health`
- Rate limit: `GET {{base_url}}/api/v1/rate-limit/status`
- Provider validate: `GET {{base_url}}/api/v1/providers/validate?provider={{provider}}`
- List models: `GET {{base_url}}/api/v1/models?provider={{provider}}&only_gemini=true`
- Chat: `POST {{base_url}}/api/v1/chat` with body `{ "question": "...", "session_id": "..." }`

Tip: If you run via Docker, ensure the backend is mapped to host port 18000 and reachable from your machine.
