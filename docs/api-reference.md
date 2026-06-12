# JGuard API Reference

This document describes the current backend API after the session-based refactor.

Base URL:

```text
/api
```

If `JGUARD_API_PREFIX` is set, the base prefix may differ.

## Overview

The backend is now session-driven:

- Create a session once.
- Send one prompt per chat turn.
- Keep history on the server side.
- Delete the session when the conversation ends.

The old stateless chat flow that sent full history on every request has been replaced.

## Session Configuration

A session is created with a config object that controls which defense stack and LLM backend will be used for the entire conversation.

### `SessionConfig`

```json
{
  "chat_mode": "foundational",
  "local_llm": true,
  "llm_api_key": "",
  "llm_type": "llama3.1",
  "obfuscation_protection": true,
  "multi_turn_protection": false,
  "roleplay_protection": false,
  "pii_protection": true,
  "pii_strategy": "mask"
}
```

Fields:

- `chat_mode`: `foundational` or `agent`
- `local_llm`: `true` for Ollama, `false` for closed-source providers
- `llm_api_key`: required when `local_llm` is `false`
- `llm_type`: model name or provider model id
- `obfuscation_protection`: enables the obfuscation guard
- `multi_turn_protection`: reserved for the multi-turn module integration
- `roleplay_protection`: reserved for future roleplay defense logic
- `pii_protection`: enables the PII detector
- `pii_strategy`: `mask`, `encrypt`, or `block`

## Endpoints

### Create Session

`POST /api/sessions`

Creates a new conversation session and initializes backend-owned state for that session.

Request body:

```json
{
  "config": {
    "chat_mode": "foundational",
    "local_llm": true,
    "llm_api_key": "",
    "llm_type": "llama3.1",
    "obfuscation_protection": true,
    "multi_turn_protection": false,
    "roleplay_protection": false,
    "pii_protection": true,
    "pii_strategy": "mask"
  }
}
```

Response body:

```json
{
  "session_id": "9c4e1f34-3d8d-4d5d-8c8d-9d1f5fdc2f14",
  "created_at": "2026-06-12T10:15:30.123456+00:00",
  "last_active": "2026-06-12T10:15:30.123456+00:00",
  "config": {
    "chat_mode": "foundational",
    "local_llm": true,
    "llm_api_key": "",
    "llm_type": "llama3.1",
    "obfuscation_protection": true,
    "multi_turn_protection": false,
    "roleplay_protection": false,
    "pii_protection": true,
    "pii_strategy": "mask"
  }
}
```

Notes:

- The session stores its own history on the backend.
- The client should keep the returned `session_id` and use it for all later chat turns.

### Send a Chat Turn

`POST /api/sessions/{session_id}/chat`

Sends a single user prompt to the active session. The backend appends the prompt to session history, runs active defenses, and returns the reply.

Request body:

```json
{
  "prompt": "Explain the difference between obfuscation and PII protection."
}
```

Response body:

```json
{
  "reply": "...",
  "blocked": false,
  "triggered_defense": null,
  "decision": null,
  "harm_label": null,
  "timestamp": "2026-06-12T10:15:35.123456+00:00"
}
```

Response fields:

- `reply`: assistant output or block message
- `blocked`: `true` if a defense stopped the request
- `triggered_defense`: `obfuscation`, `pii`, or `null`
- `decision`: obfuscation verdict when available
- `harm_label`: harm label when available
- `timestamp`: server timestamp for the turn

Behavior:

- For `chat_mode = foundational`, the backend routes the prompt through the foundational LLM path.
- For `chat_mode = agent`, the backend routes the prompt through the multi-agent system.
- The backend uses the stored session history when calling the selected model.
- The client does not send history on each turn.

### Get Session History

`GET /api/sessions/{session_id}/history`

Returns the authoritative conversation history stored by the backend.

Response body:

```json
{
  "session_id": "9c4e1f34-3d8d-4d5d-8c8d-9d1f5fdc2f14",
  "history": [
    {
      "role": "user",
      "content": "Hello"
    },
    {
      "role": "assistant",
      "content": "Hi, how can I help?"
    }
  ],
  "created_at": "2026-06-12T10:15:30.123456+00:00",
  "last_active": "2026-06-12T10:15:35.123456+00:00",
  "meta": {}
}
```

History item schema:

- `role`: `system`, `user`, or `assistant`
- `content`: message text

### Delete Session

`DELETE /api/sessions/{session_id}`

Deletes the session and its in-memory state.

Response body:

```json
{
  "success": true,
  "session_id": "9c4e1f34-3d8d-4d5d-8c8d-9d1f5fdc2f14"
}
```

## Error Responses

Common error responses are returned as FastAPI HTTP errors.

### `404 Not Found`

Returned when the session id does not exist.

```json
{
  "detail": "Session not found."
}
```

### `500 Internal Server Error`

Returned when a downstream model/runtime call fails.

```json
{
  "detail": "<runtime error message>"
}
```

## Typical Flow

1. Create a session with `POST /api/sessions`.
2. Store the returned `session_id` on the client.
3. Send one prompt at a time to `POST /api/sessions/{session_id}/chat`.
4. Query `GET /api/sessions/{session_id}/history` if you need the conversation state.
5. Delete the session with `DELETE /api/sessions/{session_id}` when finished.

## Implementation Notes

- Runtime resources such as the PII engine and agent graph are initialized once during FastAPI startup.
- Session history now lives on the backend instead of being duplicated from the frontend payload.
- The session store is currently in-memory, so sessions are process-local.
- If the backend is deployed with multiple workers, the session store should be moved to Redis or another shared store.
- The `multi_turn_protection` field is reserved for the next step, where the multi-turn detector will inspect each message incrementally inside a session.
