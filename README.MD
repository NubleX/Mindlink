## Environment
Create a `.env` in the project root (or export env vars):
```
OPENAI_API_KEY=sk-your-key
AI_PROVIDER=openai
AI_MODEL=gpt-5
AI_MEMORY_TURNS=50
```

## Usage
- One-off prompt (project-local memory):
  `mindlink --prompt "help me design a Rust scanner"`
- Interactive chat: `mindlink chat`
- Global memory (not project specific): `mindlink --no-project-memory --prompt "..."`
- Memory ops: `mindlink memory-show [N]` | `mindlink memory-clear`
