# Deployment

## 1) Requisitos
- Python 3.11 (recomendado)
- (Opcional) GPU CUDA compatible para acelerar Transformers
- Crear entorno: `conda create -n chatbot python=3.11 && conda activate chatbot`

## 2) Instalar dependencias
pip install -r requirements.txt

## 3) Configuración
1. Copia `.env.example` a `.env` y completa:
   - `APP_USER` / `APP_PASS` para auth básica de Gradio (opcional pero recomendado).
   - Rutas de KB: `KB_ES_PATH`, `KB_EN_PATH` si no usas las predeterminadas.
   - Parám. de S11 (rate-limit y truncado) si quieres ajustar.

2. Coloca `data/products.csv` con encabezados: `id,title,category,price,stock,url`.

## 4) Ejecución local
python app.py
Accede a http://127.0.0.1:7860

## 5) Producción (ideas)
- **Reverse proxy** (Nginx/Caddy) sobre `127.0.0.1:7860` con TLS.
- **Auth**: habilita `APP_USER/APP_PASS` en `.env`.
- **Logs**: rota `logs/session-*.jsonl`.
- **Windows**: NSSM/Task Scheduler para servicio.
- **Linux**: `systemd` servicio simple.

## 6) Docker (opcional, esquema)
- Crear `Dockerfile`, copiar `app.py`, `requirements.txt`, `data/` y `docs/`.
- `ENV` para `.env` o `--env-file`.
- Exponer `7860` y correr `python app.py`.

## 7) Seguridad rápida
- Mantén `.env` fuera del repo público.
- Revisa `SECURITY.md` para reportes y disclosure.
- Aplica parches S7 y S11 ya incluidos (límites CSV, URL whitelist, rate-limit, truncado, etc.).

