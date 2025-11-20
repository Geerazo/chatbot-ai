# API / Interface

Este proyecto **no expone** un API REST público; la interacción se hace a través de la interfaz **Gradio** (`type="messages"`).

## Modelo de interacción (UI)
Cada turno del chat entrega/recibe estructuras como:
```json
[
  {"role": "user", "content": "Do you ship to Canada?"},
  {"role": "assistant", "content": "Yes, we ship to Canada..."}
]
