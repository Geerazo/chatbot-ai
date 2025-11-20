# TechStore · Asistente Virtual (ES/EN)

Chatbot bilingüe para atención al cliente (envíos, devoluciones, stock) con **Qwen** + **Gradio**.  
Incluye: intents heurísticos, grounding de FAQ (RapidFuzz+YAML), catálogo CSV interno, moderación (bad words + Detoxify), logging JSONL y parches de seguridad (rate-limit, truncado, sanitización de URLs).

## Demo local
```bash
conda activate chatbot   # o tu venv
pip install -r requirements.txt
cp .env.example .env     # configura variables
python app.py