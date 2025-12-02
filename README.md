![CI](https://img.shields.io/github/actions/workflow/status/Geerazo/chatbot-ai/ci.yml?branch=main&label=CI)
![CodeQL](https://img.shields.io/github/actions/workflow/status/Geerazo/chatbot-ai/codeql.yml?branch=main&label=CodeQL)
![License](https://img.shields.io/github/license/Geerazo/chatbot-ai)

# TechStore · Asistente Virtual (ES/EN)

Asistente virtual bilingüe (español / inglés) para atención al cliente de una tienda ficticia **TechStore**.  
Responde preguntas sobre **envíos, devoluciones, estado de pedidos y stock**, integrando:

- Modelo de lenguaje (**Qwen**)
- Interfaz web con **Gradio**
- Intents heurísticos + FAQ internas
- Catálogo de productos desde CSV
- Capa de moderación y seguridad
- CI/CD, CodeQL y reglas de protección en GitHub

Este proyecto está pensado como ejemplo de **chatbot de soporte al cliente listo para producción**, con énfasis en buenas prácticas de ingeniería y seguridad.

---

## ✨ Features principales

- 🌐 **Bilingüe (ES/EN)**  
  Detección simple del idioma del usuario y respuesta coherente en español o inglés.

- 🧠 **NLP + Heurísticas**  
  - Intents heurísticos (envíos, devoluciones, stock, horarios, contacto).  
  - Grounding con FAQ internas usando similitud difusa (**RapidFuzz + YAML**).  
  - Contexto adicional sobre políticas de la tienda para evitar “alucinaciones” del modelo.

- 🛍️ **Catálogo interno de productos**  
  - Lectura de un archivo CSV con productos, precios y stock.  
  - Respuestas contextualizadas (“Sí hay stock de X”, “Solo quedan N unidades”, etc.).

- 🛡️ **Moderación y seguridad**  
  - Filtro de bad words.  
  - Análisis de toxicidad (opcional) con **Detoxify**.  
  - Rate limiting básico para evitar abuso.  
  - Truncado de prompts muy largos.  
  - Sanitización de URLs u otros inputs potencialmente peligrosos.

- 📊 **Logging estructurado**  
  - Registro de interacciones en formato **JSONL** para posteriores análisis.  
  - Campos típicos: timestamp, idioma, intent, tipo de respuesta, texto del usuario, respuesta generada.

- ⚙️ **Ingeniería y calidad**  
  - CI con **GitHub Actions**: build + tests en cada PR.  
  - Escaneo de seguridad con **CodeQL**.  
  - Reglas de protección de rama (`main`) y checks obligatorios.

---

## 🧱 Stack tecnológico

- **Backend / Lógica**: Python 3.x  
- **Modelo de lenguaje**: Qwen (vía API / cliente correspondiente)  
- **Interfaz**: [Gradio](https://www.gradio.app/)  
- **Similitud / Grounding**: [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) + archivos YAML  
- **Moderación**: “bad words list” + [Detoxify](https://github.com/unitaryai/detoxify) (opcional)  
- **Persistencia simple**: CSV (catálogo) + JSONL (logs)  
- **CI/CD**: GitHub Actions (`ci.yml`)  
- **Seguridad estática**: GitHub CodeQL (`codeql.yml`)

---

## 🚀 Cómo ejecutar en local

### 1. Clonar el repositorio

git clone https://github.com/Geerazo/chatbot-ai.git
cd chatbot-ai

2. Crear entorno e instalar dependencias

Con venv:
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
(si usas conda, puedes añadir un bloque alternativo aquí)

3. Configurar variables de entorno

cp .env.example .env
Edita .env con tus credenciales (API key de Qwen u otro proveedor, configuración de logs, flags de moderación, etc.).

4. Ejecutar el chatbot

python app.py
Por defecto, Gradio levantará una interfaz web en http://127.0.0.1:7860 (o similar).
Abre esa URL en tu navegador y comienza a conversar con el asistente.

🗂️ Estructura del proyecto (simplificada)


chatbot-ai/
├─ app.py                # Punto de entrada del chatbot (Gradio + lógica)
├─ README.md             # Documentación del proyecto
├─ requirements.txt      # Dependencias de Python
├─ .env.example          # Ejemplo de configuración de entorno
├─ data/
│   ├─ catalog.csv       # Catálogo de productos (ejemplo)
│   └─ faq.yml           # FAQ internas con respuestas canónicas
├─ logs/
│   └─ interactions.jsonl  # Logs estructurados (generados en runtime)
├─ .github/
│   └─ workflows/
│       ├─ ci.yml        # CI: build + test
│       └─ codeql.yml    # Análisis estático de seguridad
└─ ...

🔒 Seguridad

Resumen de medidas de seguridad implementadas:

Validación básica del input del usuario.

Lista de palabras prohibidas + análisis de toxicidad (Detoxify) para bloquear contenido ofensivo.

Rate limiting para evitar abuso.

Truncado de prompts demasiado largos.

Sanitización de URLs u otros campos de entrada.

Escaneo de vulnerabilidades en el código con CodeQL (GitHub Security).

Reglas de protección para la rama main (sin merge commits directos, PR + CI obligatorio).

⚠️ Aunque este proyecto incorpora varias prácticas de seguridad, no sustituye una auditoría de seguridad completa para entornos de producción críticos.


🧭 Roadmap (ideas futuras)

Integrar base de conocimientos vectorial (RAG) para políticas más complejas.

Añadir autenticación básica para panel interno de administración.

Extender métricas: tiempos de respuesta, satisfacción aproximada, tasa de fallback al modelo.

Integrar trazas y monitoring (Prometheus/Grafana o similar).


🤝 Contribuciones

Actualmente el proyecto se mantiene como parte de mi portafolio técnico, pero se agradecen:

Issues con bugs o mejoras sugeridas.

Pull Requests pequeños y bien descritos.


📄 Licencia

Este proyecto se distribuye bajo la licencia especificada en el archivo LICENSE de este repositorio.



