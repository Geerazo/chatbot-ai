# Architecture

## Overview
**TechStore Assistant** is a bilingual (ES/EN) customer-support chatbot built with:
- **Transformers (Qwen Instruct)** for generation
- **Gradio (Blocks)** UI (`type="messages"`)
- **Heuristic intents** (shipping / returns / stock)
- **FAQ grounding** with RapidFuzz + YAML
- **Catalog RAG** from `data/products.csv`
- **Moderation** (bad-words + optional Detoxify)
- **Logging** JSONL por sesión
- **Hardening**: input truncation, rate-limit, URL sanitization, system-override sanitization, CSV reader limits

## High-Level Diagram

[User/Gradio UI] ──> [Wrapper S11: trunc + rate-limit + lang resolve]
│
├─> [Intents S8] -> (shipping/returns) -> deterministic reply
│
├─> [Stock intent] -> [Catalog RAG S7] -> format_hits
│
├─> [FAQ S9] -> (direct|context) -> system override (sanitized)
│
└─> [LLM S6 + S5 prompts]
└─> [Moderation + Refinement]


## Key Modules
- **S5 Prompts & Language:** anti-mix (memoria de idioma), plantillas y refinado.
- **S6 Generation:** `apply_chat_template` + `generate` con restricciones (bad_words_ids, no-repeat, beam search).
- **S7 Catalog RAG:** parsing robusto, CSV sniffer, límites de filas, whitelist de URLs, búsqueda y formateo seguro.
- **S11 Wrapper:** truncado, rate limiting y normalización de historial.

## Configuration
- Variables `.env` (`APP_USER`, `APP_PASS`, `KB_ES_PATH`, `KB_EN_PATH`, límites S11, etc.)
- `requirements.txt` con versiones compatibles (Windows CPU/GPU).
