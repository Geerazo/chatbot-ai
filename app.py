# ============================================================
# Índice de Secciones (TOC) – usar los IDs [Sx] y [Sx.y] para parches
# ============================================================
# [S0]  Imports y configuración base 
# [S1]  Metadatos del proyecto
# [S2]  Modelos / Carga de tokenizers y pesos
# [S3]  Moderación (bad words + Detoxify)
# [S4]  Logging JSONL por sesión
# [S5]  Prompts y utilidades de idioma
# --- [S5.1] Detección de idioma robusta
# --- [S5.2] Resolución de idioma con memoria (anti-mezcla)
# --- [S5.3] Construcción de mensajes (apply_chat_template)
# --- [S5.4] Refinado de salida (conciso + follow-up)
# [S6]  Núcleo de decodificación (encode + generate)
# [S7] Catalog RAG & Actions
# --- [S7.1] Utilidades de parsing para dinero
# --- [S7.2] Modelo de producto
# --- [S7.3] CSV sniffer (lector + headers normalizados)
# --- [S7.4] Carga de catálogo (con hardening
# --- [S7.5] Carga inicial y flag de disponibilidad
# --- [S7.6] Normalización de términos de búsqueda
# --- [S7.7] Sanitización de URLs (whitelist http/https)
# --- [S7.8] Búsqueda en catálogo
# --- [S7.9] Formateo de resultados (seguro)
# --- [S7.10] Acción para botón “Laptops <$1200”
# [S8]  Intents y Slots (heurístico ligero)
# [S9]  FAQ Grounding (RapidFuzz + YAML) con rutas robustas
# [S10]  Motor de respuesta (orden: Intents → FAQ → LLM)
# [S11] Wrapper Gradio (type="messages")
# [S12] Quick Eval (herramienta dev)
# [S13] Interfaz de usuario Gradio (Blocks UI)
# ============================================================


# =======================
# [S0]  Imports y configuración base
# =======================
import os
import re
import time
import uuid
import json
import torch
from pathlib import Path
from datetime import datetime
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
import gradio as gr
from dataclasses import dataclass
import csv
from typing import List, Tuple, Optional
from rapidfuzz import process, fuzz
import yaml

hf_logging.set_verbosity_error()
BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
CATALOG_CSV = str(DATA_DIR / "products.csv")
# =============================================
# [S1] Metadatos del proyecto (informativo)
# =============================================
# Chatbot bilingüe (ES/EN) con Qwen Instruct + Gradio (Blocks)
# - Selector de idioma (Auto/ES/EN) con memoria de idioma (parche anti mezclas)
# - Chat template nativo (apply_chat_template)
# - Respuestas concisas + 1 follow-up
# - Moderación (bad words + Detoxify), antibucle y refinado
# - FAQ Grounding (RapidFuzz + YAML) con rutas robustas
# - Logging JSONL por sesión + Quick Eval (dev)


# ============================================
# [S2] Modelos / Carga de tokenizers y pesos
# ============================================
MODEL_NAME_EN = os.getenv("MODEL_NAME_EN", "Qwen/Qwen2.5-1.5B-Instruct")
MODEL_NAME_ES = os.getenv("MODEL_NAME_ES", "Qwen/Qwen2.5-1.5B-Instruct")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name: str):
    trust = os.getenv("HF_TRUST_REMOTE_CODE", "0") == "1"
    local_only = os.getenv("HF_LOCAL_FILES_ONLY", "0") == "1"
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust,
        local_files_only=local_only,
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust,
        local_files_only=local_only,
    )
    mdl.to(DEVICE)
    mdl.eval()
    return tok, mdl

tokenizer_en, model_en = load_model_and_tokenizer(MODEL_NAME_EN)
try:
    tokenizer_es, model_es = load_model_and_tokenizer(MODEL_NAME_ES)
except Exception as e:
    print(f"[WARN] Modelo ES '{MODEL_NAME_ES}' no cargó: {e} -> usando EN como fallback")
    tokenizer_es, model_es = tokenizer_en, model_en


# ============================================
# [S3] Moderación (bad words + Detoxify)
# ============================================
BAD_WORDS = [
    "mierda","cagada","idiota","imbécil","estúpido","puto","puta","pendejo","pendeja","gilipollas",
    "hijo de puta","hpta","hp","malparido","marica","gonorrea",
    "fuck","shit","bitch","asshole","moron","idiot","stupid","dumb","bastard","retard"
]

def build_bad_words_ids(tokenizer, bad_words):
    ids = []
    for w in bad_words:
        try:
            toks = tokenizer.encode(w, add_special_tokens=False)
            if toks:
                ids.append(toks)
        except Exception:
            pass
    return ids

bad_words_ids_en = build_bad_words_ids(tokenizer_en, BAD_WORDS)
bad_words_ids_es = build_bad_words_ids(tokenizer_es, BAD_WORDS)

# Detoxify (lazy)
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
_TOX_MODEL = None
_TOX_MODEL_NAME = os.getenv("DETOXIFY_MODEL", "original")

def get_tox():
    global _TOX_MODEL
    if _TOX_MODEL is not None:
        return _TOX_MODEL
    try:
        from detoxify import Detoxify  # type: ignore
        _TOX_MODEL = Detoxify(_TOX_MODEL_NAME, device="cpu")
        return _TOX_MODEL
    except Exception as e:
        print(f"[WARN] Detoxify no disponible ({_TOX_MODEL_NAME}): {e}")
        return None


# ============================================
# [S4] Logging JSONL por sesión
# ============================================
LOG_ENABLED = os.getenv("LOG_ENABLED", "1") == "1"
LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SESSION_ID = os.getenv("SESSION_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
LOG_FILE = LOGS_DIR / f"session-{SESSION_ID}.jsonl"

_PII_PATTERNS = [
    (re.compile(r"[A-Z0-9]{2,}-?\d{2,}-?[A-Z0-9]{2,}", re.I), "<ORDER>"),
    (re.compile(r"\b\d{6,}\b"), "<NUM>"),
    (re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I), "<EMAIL>"),
    (re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d"), "<PHONE>")
]
def _redact(obj):
    try:
        s = json.dumps(obj, ensure_ascii=False)
        for rx, rep in _PII_PATTERNS: s = rx.sub(rep, s)
        return json.loads(s)
    except Exception:
        return obj

def log_event(event: str, **payload):
    if not LOG_ENABLED:
        return
    record = _redact({
    "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
    "session": SESSION_ID,
    "event": event,
    **payload
})
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] log_event failed: {e}")


# ============================================
# [S5] Prompts y utilidades de idioma
# ============================================
SYSTEM_ES = (
    "Eres el asistente virtual de TechStore. Responde en español claro y profesional. "
    "Primero da una respuesta directa en 1–2 oraciones. Luego, si corresponde, añade UNA sola pregunta de seguimiento. "
    "Sé específico para envíos, devoluciones o stock."
)
SYSTEM_EN = (
    "You are TechStore’s virtual assistant. Answer in concise, professional English. "
    "First give a direct answer in 1–2 sentences. Then, if relevant, add ONE follow-up question. "
    "Be specific for shipping, returns or stock."
)
SYSTEM_ES_STRICT = (
    "Eres el asistente virtual de TechStore. RESPONDE ÚNICAMENTE EN ESPAÑOL NEUTRO. "
    "Sé conciso (1–2 oraciones) y, si corresponde, añade UNA sola pregunta de seguimiento. "
    "No cambies de idioma bajo ninguna circunstancia."
)
SYSTEM_EN_STRICT = (
    "You are TechStore’s virtual assistant. RESPOND ONLY IN ENGLISH. "
    "Be concise (1–2 sentences) and add ONE follow-up question if relevant. "
    "Do not switch languages under any circumstance."
)

# [S5.1] Detección segura
def _detect_lang_safe(text: str) -> str:
    try:
        return "es" if detect(text).startswith("es") else "en"
    except Exception:
        return "es"

# [S5.2] Resolución de idioma con memoria (anti-mezcla)
def resolve_lang(user_text: str, pref: str, history_pairs: list[tuple[str, str]]) -> str:
    """
    - Si pref ≠ auto -> usar pref fijo.
    - En auto:
       * Si el texto es corto (<5 chars), hereda el idioma del último assistant o user del historial.
       * Si no hay historial útil, usar detección; si falla, caer en 'es'.
    """
    if pref in ("es", "en"):
        return pref

    txt = (user_text or "").strip()
    if len(txt) < 5:
        for u, b in reversed(history_pairs or []):
            if b and b.strip():
                return _detect_lang_safe(b)
        for u, b in reversed(history_pairs or []):
            if u and u.strip():
                return _detect_lang_safe(u)
        return "es"
    return _detect_lang_safe(txt)

# [S5.3] Construcción de mensajes
def build_chat_messages(lang: str, history_pairs: list[tuple[str, str]], user_msg: str, system_override: str | None = None):
    system = system_override or (SYSTEM_ES if lang == "es" else SYSTEM_EN)
    msgs = [{"role": "system", "content": system}]
    if system_override is None:
        if lang == "es":
            msgs += [
                {"role": "user", "content": "¿Hacen envíos internacionales?"},
                {"role": "assistant", "content": "Sí, realizamos envíos internacionales con tiempos típicos de 2–5 días hábiles. "
                                                 "¿En qué ciudad/país te encuentras?"},
            ]
        else:
            msgs += [
                {"role": "user", "content": "Do you offer returns?"},
                {"role": "assistant", "content": "Yes, returns are accepted within 30 days in original packaging with proof of purchase. "
                                                 "Would you like me to walk you through the steps?"},
            ]
    for u, b in (history_pairs or [])[-3:]:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": b})
    msgs.append({"role": "user", "content": user_msg})
    return msgs

# [S5.4] Refinado de salida
def _refine_concise(text: str, lang: str) -> str:
    if len(text) > 1000:
        text = text[:1000]
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s for s in sents if s]
    core = " ".join(sents[:2]) if len(sents) >= 2 else (sents[0] if sents else text)
    keywords = ["envío","envios","entrega","devolución","stock","shipping","return","refund","available","in stock"]
    if any(k in text.lower() for k in keywords):
        follow = "¿Deseas que te indique los pasos?" if lang == "es" else "Would you like me to outline the steps?"
        return f"{core} {follow}"
    return core


# ============================================
# [S6] Núcleo de decodificación (generate)
# ============================================
def _encode_and_generate(tok, mdl, chat_messages, bad_ids, eos_id):
    try:
        prompt_text = tok.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_text = "\n".join([m["content"] for m in chat_messages])

    enc = tok(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=False,
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    gen_kwargs = dict(
        max_new_tokens=96,
        do_sample=False,
        num_beams=4,
        early_stopping=True,
        length_penalty=0.9,
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        bad_words_ids=bad_ids
    )

    output_ids = mdl.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs
    )
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

# ===============================
# [S7] Catalog RAG & Actions
# ===============================

# --- [S7.1] Utilidades de parsing para dinero -------------------------------
_NUM_RE = re.compile(r"[-+]?\d[\d.,]{0,20}")

def _to_float(text: str | None) -> float | None:
    if not text:
        return None
    m = _NUM_RE.search(text)
    if not m:
        return None
    raw = m.group(0).replace("\u202f", "").replace(" ", "")
    has_dot, has_comma = "." in raw, "," in raw

    if has_dot and has_comma:
        last_dot, last_comma = raw.rfind("."), raw.rfind(",")
        if last_comma > last_dot:
            # 1.299,50 -> 1299.50
            try:
                return float(raw.replace(".", "").replace(",", "."))
            except ValueError:
                return None
        else:
            # 1,299.50 -> 1299.50
            try:
                return float(raw.replace(",", ""))
            except ValueError:
                return None

    if has_comma and not has_dot:
        if raw.count(",") == 1:
            left, right = raw.split(",")
            if right.isdigit() and len(right) in (1, 2):
                try:
                    return float(f"{left}.{right}")
                except ValueError:
                    return None
        try:
            return float(raw.replace(",", ""))
        except ValueError:
            return None

    try:
        return float(raw)
    except ValueError:
        return None


# --- [S7.2] Modelo de producto ---------------------------------------------
@dataclass
class Product:
    id: str
    title: str
    category: str
    price: float
    stock: int
    url: str


# --- [S7.3] CSV sniffer (lector + headers normalizados) ---------------------
def _sniff_reader_and_headers(f):
    """Devuelve (csv.DictReader, headers_normalizados). Soporta ',' y ';'."""
    sample = f.read(2048)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        reader = csv.DictReader(f, dialect=dialect)
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        return reader, headers
    except Exception:
        # Fallback fijo a coma
        class _D(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        reader = csv.DictReader(f, dialect=_D())
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        return reader, headers


# --- [S7.4] Carga de catálogo (con hardening) -------------------------------
def _coerce_float(x, default=0.0):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default

def _coerce_int(x, default=0):
    try:
        return int(float(str(x).replace(",", ".")))
    except Exception:
        return default

def _load_catalog(path: str | Path | None = None) -> list[Product]:
    """
    Carga catálogo desde CSV con:
    - fallback de ruta si no existe CATALOG_CSV
    - límite MAX_ROWS para evitar abuso/memoria
    """
    if path is None:
        try:
            path = CATALOG_CSV  # definido en [S0]
        except NameError:
            path = str((Path(__file__).parent / "data" / "products.csv"))

    path = Path(path)
    print(f"[CATALOG] looking for CSV at: {path}")
    if not path.exists():
        print(f"[CATALOG] file not found at {path}")
        return []

    items: list[Product] = []
    MAX_ROWS = 5000  # [hardening] límite defensivo de filas

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader, headers = _sniff_reader_and_headers(f)
        need = {"id", "title", "category", "price", "stock", "url"}
        if not need.issubset(set(headers)):
            print(f"[CATALOG] bad headers: {headers} (need {sorted(need)})")

        for i, row in enumerate(reader):
            if i >= MAX_ROWS:
                break  # [hardening] corte seguro
            try:
                r = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
                p = Product(
                    id=r.get("id", ""),
                    title=r.get("title", ""),
                    category=r.get("category", ""),
                    price=_coerce_float(r.get("price", "0")),
                    stock=_coerce_int(r.get("stock", "0")),
                    url=r.get("url", ""),
                )
                if p.id and p.title:
                    items.append(p)
            except Exception:
                continue
    print(f"[CATALOG] loaded {len(items)} items from {path}")
    return items


# --- [S7.5] Carga inicial y flag de disponibilidad --------------------------
try:
    _CATALOG: list[Product] = _load_catalog()
except Exception as _e:
    print("[CATALOG] load failed:", _e)
    _CATALOG = []
CATALOG = len(_CATALOG) > 0


# --- [S7.6] Normalización de términos de búsqueda --------------------------
def _normalize_terms(query: str | None) -> list[str]:
    if not query:
        return []
    q = query.lower().strip()
    syn = {
        "laptops": "laptop",
        "notebooks": "laptop",
        "notebook": "laptop",
        "portatil": "laptop",
        "portátiles": "laptop",
        "portatiles": "laptop",
        "gaming": "gamer",
    }
    raw_terms = re.findall(r"[a-záéíóúñ0-9\-]{1,50}", q)
    terms = []
    for t in raw_terms:
        t2 = syn.get(t, t)
        if t2.endswith("s") and len(t2) > 3:
            terms += [t2, t2[:-1]]
        else:
            terms.append(t2)
    if any(x in ("laptop","notebook","notebooks","laptops","portatil","portátiles","portatiles") for x in raw_terms):
        terms.append("laptop")
    seen, norm = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t); norm.append(t)
    return norm


# --- [S7.7] Sanitización de URLs (whitelist http/https) ---------------------
_ALLOWED_SCHEMES = ("http://", "https://")

def _safe_url(u: str) -> str:
    u = (u or "").strip()
    if not u.lower().startswith(_ALLOWED_SCHEMES):
        return ""
    try:
        from urllib.parse import urlparse
        p = urlparse(u)
        return u if p.scheme in ("http", "https") and p.netloc else ""
    except Exception:
        return ""


# --- [S7.8] Búsqueda en catálogo -------------------------------------------
def catalog_search(query: str | None = None, budget: float | None = None, top_k: int = 3) -> list[Product]:
    pool = _CATALOG
    if budget is not None:
        pool = [p for p in pool if p.price <= budget]
    terms = _normalize_terms(query)
    if terms:
        filt = []
        for p in pool:
            tl, cl = p.title.lower(), p.category.lower()
            if any(t in tl or t in cl for t in terms):
                filt.append(p)
        pool = filt if filt else pool
    pool = sorted(pool, key=lambda p: p.price)
    return pool[:top_k]


# --- [S7.9] Formateo de resultados (seguro) --------------------------------
def format_hits(hits: list[Product], lang: str) -> str:
    if not hits:
        return "No hay resultados en catálogo." if lang == "es" else "No results found in catalog."
    bullet = []
    for p in hits:
        link = _safe_url(p.url)  # [seguridad] sanitiza la URL
        if lang == "es":
            bullet.append(f"• {p.title} — ${int(p.price)} — Stock: {p.stock}" + (f" — {link}" if link else ""))
        else:
            bullet.append(f"• {p.title} — ${int(p.price)} — Stock: {p.stock}" + (f" — {link}" if link else ""))
    follow = "¿Deseas ver más resultados o filtrar por marca?" if lang == "es" \
             else "Would you like to see more results or filter by brand?"
    head = "Opciones disponibles:" if lang == "es" else "Available options:"
    return head + "\n" + "\n".join(bullet) + f"\n{follow}"


# --- [S7.10] Acción para botón “Laptops <$1200” -----------------------------
def action_laptops_under(messages, pref_lang, max_price: str = "1200"):
    """
    Devuelve SOLO 'messages' con una consulta de usuario inyectada.
    Luego [S13] encadena con bot_reply para responder.
    """
    if isinstance(pref_lang, str) and pref_lang == "en":
        utxt = f"I’m looking for a laptop under ${max_price}"
    elif isinstance(pref_lang, str) and pref_lang == "es":
        utxt = f"Busco una laptop por debajo de ${max_price}"
    else:
        utxt = f"Laptop under ${max_price}"

    if messages is None:
        messages = []
    messages = list(messages) + [{"role": "user", "content": utxt}]
    return messages


# ============================================
# [S8] Intents y Slots (heurístico ligero)
# ============================================
INTENT_KEYWORDS = {
    "shipping": {
        "es": ["envio", "envío", "envíos", "enviar", "entrega", "entregan", "tiempo de entrega", "costo de envio", "enviar a"],
        "en": ["ship", "shipping", "deliver", "delivery", "international", "cost to", "send to"]
    },
    "returns": {
        "es": ["devolución", "devolver", "reembolso", "cambio", "garantía"],
        "en": ["return", "refund", "exchange", "warranty", "rma"]
    },
    "stock": {
        "es": ["stock", "disponible", "disponibilidad", "tienen", "hay", "existencias"],
        "en": ["stock", "available", "availability", "have in stock", "in stock"]
    }
}

COMMON_COUNTRIES = [
    "mexico","méxico","colombia","peru","perú","chile","argentina","spain","españa","usa","united states","canada","canadá",
    "uk","united kingdom","france","germany","italy","brazil","brasil","ecuador","uruguay","paraguay","bolivia",
]

def detect_intent(text: str, lang: str) -> str | None:
    t = text.lower()
    if re.search(r"(\$|usd|cop|mxn|eur|€)\s?[\d\.,]{2,}", t) or any(x in t for x in ["<", "under", "menos de", "debajo de"]):
        return "stock"
    best, best_hits = None, 0
    for intent, lex in INTENT_KEYWORDS.items():
        keys = lex["es"] if lang == "es" else lex["en"]
        hits = sum(1 for k in keys if k in t)
        if hits > best_hits:
            best, best_hits = intent, hits
    return best if best_hits >= 1 else None


def extract_slots(intent: str, text: str, lang: str) -> dict:
    t = text.strip()
    tl = t.lower()
    slots = {}

    if intent == "shipping":
        found_country = None
        for c in COMMON_COUNTRIES:
            if c in tl:
                found_country = c
                break
        slots["country"] = found_country
        import re as _re
        m = _re.search(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})\b", t)
        slots["city"] = m.group(1) if m else None

    elif intent == "returns":
        import re as _re
        if len(tl) > 200: 
            m = None
        else:
            m = _re.search(r"(pedido|orden|order)\s{0,10}#?\s{0,10}([A-Z0-9\-]{5,20})", tl, flags=_re.IGNORECASE)
        slots["order_id"] = m.group(2).upper() if m else None

    elif intent == "stock":
        import re as _re
        money = _re.search(r"(\$|usd|cop|mxn|€)\s?([\d\.,]{2,})", tl)
        slots["budget"] = money.group(0) if money else None
        generic = {"stock","disponible","availability","available","tienen","hay","in","en","de","the","la","el","un","una"}
        words = [w for w in re.findall(r"[A-Za-z0-9\-]+", t) if len(w) > 2]
        prod = [w for w in words if w.lower() not in generic][:3]
        slots["product"] = " ".join(prod) if prod else None

    return slots

def missing_slots(intent: str, slots: dict, lang: str) -> list[str]:
    missing = []
    if intent == "shipping":
        if not slots.get("country") and not slots.get("city"):
            missing.append("country_city")
        elif not slots.get("country"):
            missing.append("country")
        elif not slots.get("city"):
            missing.append("city")
    elif intent == "returns":
        if not slots.get("order_id"):
            missing.append("order_id")
    elif intent == "stock":
        if not slots.get("product") and not slots.get("budget"):
            missing.append("product_or_budget")
    return missing

def ask_for_missing(intent: str, missing: list[str], lang: str) -> str:
    if not missing:
        return ""
    if lang == "es":
        mapping = {
            "country_city": "¿A qué país o ciudad debemos enviar?",
            "country": "¿A qué país debemos enviar?",
            "city": "¿A qué ciudad debemos enviar?",
            "order_id": "¿Podrías indicarme tu número de pedido para iniciar la devolución?",
            "product_or_budget": "¿Qué producto te interesa o qué presupuesto tienes? (ej.: “laptop gamer” o “$600 USD”)",
        }
    else:
        mapping = {
            "country_city": "Which country or city should we ship to?",
            "country": "Which country should we ship to?",
            "city": "Which city should we ship to?",
            "order_id": "Could you share your order number to start the return?",
            "product_or_budget": "Which product are you looking for, or what’s your budget? (e.g., “gaming laptop” or “$600 USD”)",
        }
    return mapping[missing[0]]

def handle_intent(intent: str, slots: dict, lang: str) -> str:
    if intent == "shipping":
        dest = slots.get("city") or slots.get("country") or ("tu ubicación" if lang=="es" else "your location")
        if lang == "es":
            return (f"Realizamos envíos a {dest}. El tiempo típico es 3–7 días hábiles y el costo depende del destino y el peso. "
                    f"¿Deseas que te indique los pasos para cotizar?")
        else:
            return (f"We ship to {dest}. Typical delivery time is 3–7 business days and cost depends on destination and weight. "
                    f"Would you like me to outline the steps to get a quote?")
    elif intent == "returns":
        oid = slots.get("order_id")
        if lang == "es":
            if oid:
                return (f"Perfecto, iniciaremos el proceso para el pedido {oid}. "
                        "Te enviaré una etiqueta de devolución y pasos de embalaje. ¿Confirmas?")
            else:
                return "Para iniciar la devolución necesito tu número de pedido. ¿Podrías compartirlo?"
        else:
            if oid:
                return (f"Great, we’ll start the return for order {oid}. "
                        "I’ll send you a return label and packing steps. Do you confirm?")
            else:
                return "To start the return I need your order number. Could you share it?"
    elif intent == "stock":
        prod, budget = slots.get("product"), slots.get("budget")
        if lang == "es":
            if prod:
                return (f"Puedo verificar disponibilidad para “{prod}”. "
                        "¿Deseas que compruebe modelos similares y tiempos de entrega?")
            if budget:
                return (f"Con un presupuesto de {budget}, puedo sugerir opciones en stock. "
                        "¿Te interesan ultralivianas, gaming o trabajo general?")
            return "¿Qué producto te interesa o qué presupuesto tienes?"
        else:
            if prod:
                return (f"I can check availability for “{prod}”. "
                        "Would you like me to include similar models and delivery times?")
            if budget:
                return (f"With a budget around {budget}, I can suggest in-stock options. "
                        "Interested in ultralight, gaming or general-purpose?")
            return "Which product are you looking for or what’s your budget?"
    return ""

def merge_slot_memory(state: Optional[dict], intent: str, new_slots: dict) -> dict:
    """
    Combina el estado previo con nuevos slots detectados.
    Si cambia la intención, resetea los slots.
    """
    state = state or {}
    if state.get("intent") != intent:
        state = {"intent": intent, "slots": {}}
    state["slots"].update({k: v for k, v in (new_slots or {}).items() if v})
    return state

def is_fulfilled(intent: str, slots: dict) -> bool:
    """
    ¿Hay información suficiente para responder determinísticamente?
    - shipping: país o ciudad
    - returns:  order_id
    - stock:    producto o presupuesto
    """
    slots = slots or {}
    if intent == "shipping":
        return bool(slots.get("country") or slots.get("city"))
    if intent == "returns":
        return bool(slots.get("order_id"))
    if intent == "stock":
        return bool(slots.get("product") or slots.get("budget"))
    return False

# ==================================================
# [S9] FAQ Grounding (RapidFuzz + YAML) rutas robustas
# ==================================================
@dataclass
class QAItem:
    q: str
    a: str

@dataclass
class KB:
    items: List[QAItem]

    @classmethod
    def load(cls, path: str) -> "KB":
        if not os.path.exists(path):
            print(f"[WARN] KB no encontrada: {path}")
            return cls([])
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        items = [QAItem(q=str(x.get("q","")).strip(), a=str(x.get("a","")).strip()) for x in data if x]
        return cls(items)

    def best_match(self, query: str) -> Tuple[Optional[QAItem], float]:
        if not self.items or not query.strip():
            return None, 0.0
        choices = [it.q for it in self.items]
        match = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)
        if not match:
            return None, 0.0
        idx = match[2]
        score = float(match[1])
        return self.items[idx], score

# Rutas de KB robustas (al lado del script)
BASE_DIR = Path(__file__).parent
KB_ES_PATH = os.getenv("KB_ES_PATH", str(BASE_DIR / "kbfaqses.yaml"))
KB_EN_PATH = os.getenv("KB_EN_PATH", str(BASE_DIR / "kbfaqsen.yaml"))
KB_ES = KB.load(KB_ES_PATH)
KB_EN = KB.load(KB_EN_PATH)

FAQ_DIRECT_THRESHOLD = float(os.getenv("FAQ_DIRECT_THRESHOLD", 88))
FAQ_CONTEXT_THRESHOLD = float(os.getenv("FAQ_CONTEXT_THRESHOLD", 78))

def faq_retrieve(lang: str, user_msg: str):
    kb = KB_ES if lang == "es" else KB_EN
    item, score = kb.best_match(user_msg)
    if not item:
        return None, None
    if score >= FAQ_DIRECT_THRESHOLD:
        return "direct", item.a
    if score >= FAQ_CONTEXT_THRESHOLD:
        prefix = "Contexto:" if lang == "es" else "Context:"
        return "context", f"{prefix} {item.a}"
    return None, None


# ======================================================
# [S10] Motor de respuesta (Intents → FAQ → LLM)
# ======================================================
@torch.inference_mode()
def generate_reply(user_message: str,
                   history_pairs: list[tuple[str, str]] | None,
                   lang: str,
                   pref_lang: str,
                   conv_state: dict | None = None):
    """
    Ahora acepta y devuelve conv_state para memoria ligera (p. ej., último idioma).
    Flujo:
      1) Intents (shipping/returns/stock) → catálogo si 'stock'
      2) FAQ grounding
      3) LLM (Qwen) con controles
    """
    conv_state = conv_state or {}
    history_pairs = history_pairs or []

    # 1) INTENT determinístico
    intent = detect_intent(user_message, lang)
    if intent:
        slots = extract_slots(intent, user_message, lang)
        miss = missing_slots(intent, slots, lang)
        log_event("intent", lang=lang, intent=intent, slots=slots, missing=miss, text=user_message)

        if intent == "stock":
            if miss:
                text = ask_for_missing(intent, miss, lang)
                conv_state["last_lang"] = lang
                return text, conv_state

            prod = slots.get("product")
            budget_raw = slots.get("budget")
            budget_val = _to_float(budget_raw) if budget_raw else None

            if CATALOG:
                hits = catalog_search(query=prod or "laptop", budget=budget_val, top_k=3)
                if hits:
                    text = format_hits(hits, lang)
                    log_event("catalog_hits", q=prod, budget=budget_val, n=len(hits))
                    conv_state["last_lang"] = lang
                    return text, conv_state

            # Fallback sin catálogo / sin resultados
            if lang == "es":
                if prod:
                    text = (f"Puedo verificar disponibilidad para “{prod}”. "
                            "¿Deseas que compruebe modelos similares y tiempos de entrega?")
                elif budget_raw:
                    text = (f"Con un presupuesto de {budget_raw}, puedo sugerir opciones en stock. "
                            "¿Te interesan ultralivianas, gaming o trabajo general?")
                else:
                    text = "¿Qué producto te interesa o qué presupuesto tienes?"
            else:
                if prod:
                    text = (f"I can check availability for “{prod}”. "
                            "Would you like me to include similar models and delivery times?")
                elif budget_raw:
                    text = (f"With a budget around {budget_raw}, I can suggest in-stock options. "
                            "Interested in ultralight, gaming or general-purpose?")
                else:
                    text = "Which product are you looking for or what’s your budget?"
            conv_state["last_lang"] = lang
            return text, conv_state

        # Otros intents (shipping/returns)
        if miss:
            text = ask_for_missing(intent, miss, lang)
            conv_state["last_lang"] = lang
            return text, conv_state
        else:
            text = handle_intent(intent, slots, lang)
            conv_state["last_lang"] = lang
            return text, conv_state

    # 2) FAQ grounding
    faq_mode, faq_text = faq_retrieve(lang, user_message)
    log_event("faq", lang=lang, mode=faq_mode or "none")
    if faq_mode == "direct":
        conv_state["last_lang"] = lang
        return faq_text, conv_state

       # 3) LLM (Qwen)
    if lang == "es":
        tok, mdl = tokenizer_es, model_es
        bad_ids = bad_words_ids_es
    else:
        tok, mdl = tokenizer_en, model_en
        bad_ids = bad_words_ids_en

    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids(tok.eos_token or "</s>")

    # --- endurecimiento para system_override a partir de FAQ ----------------
    def _sanitize_system(text: str, max_len: int = 800) -> str:
        """
        Limpia el texto que irá al 'system' para evitar prompt injection:
        - recorta longitud
        - elimina controles obvios y artefactos de plantillas
        - normaliza espacios
        """
        if not text:
            return ""
        t = text.strip()

        # recorte defensivo
        if len(t) > max_len:
            t = t[:max_len]

        # elimina caracteres/patrones comunes de inyección o markup
        t = re.sub(r"[`{}<>@]|</?script[^>]{0,50}>", " ", t, flags=re.IGNORECASE)
        t = t.replace("\x00", " ").replace("\r", " ")
        # evita prefijos tipo "SYSTEM:" "ASSISTANT:" dentro del contexto
        t = re.sub(r"\b(system|assistant|user)\s*:\s*", " ", t, flags=re.IGNORECASE)

        # colapsa espacios
        t = re.sub(r"\s+", " ", t).strip()
        return t

    system_override = None
    if faq_mode == "context" and faq_text:
        safe_ctx = _sanitize_system(faq_text)
        base = SYSTEM_ES if lang == "es" else SYSTEM_EN
        system_override = f"{base} {safe_ctx}".strip()

    # --- generación ----------------------------------------------------------
    if pref_lang in ("es", "en"):
        # Modo estricto si el usuario fijó idioma
        strict_system = SYSTEM_ES_STRICT if lang == "es" else SYSTEM_EN_STRICT
        chat_messages = build_chat_messages(
            lang, history_pairs, user_message, system_override=strict_system
        )
        t0 = time.perf_counter()
        text = _encode_and_generate(tok, mdl, chat_messages, bad_ids, eos_id)
        log_event("gen", lang=lang, strict=True, ms=int((time.perf_counter() - t0) * 1000))
    else:
        # Auto: primero con system normal (posible context FAQ seguro)
        chat_messages = build_chat_messages(
            lang, history_pairs, user_message, system_override=system_override
        )
        t0 = time.perf_counter()
        text = _encode_and_generate(tok, mdl, chat_messages, bad_ids, eos_id)
        log_event("gen", lang=lang, strict=False, attempt=1, ms=int((time.perf_counter() - t0) * 1000))

        # Verificación de deriva de idioma y reintento estricto si hace falta
        out_lang = _detect_lang_safe(text)
        drift = (out_lang != lang)
        log_event("lang_drift", expected=lang, got=out_lang, drift=drift)

        if drift:
            strict_system = SYSTEM_ES_STRICT if lang == "es" else SYSTEM_EN_STRICT
            chat_messages_strict = build_chat_messages(
                lang, history_pairs, user_message, system_override=strict_system
            )
            t0 = time.perf_counter()
            text = _encode_and_generate(tok, mdl, chat_messages_strict, bad_ids, eos_id)
            log_event("gen", lang=lang, strict=True, attempt=2, ms=int((time.perf_counter() - t0) * 1000))


    # Antibucle
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        pivot = lines[0]
        uniq = [ln for ln in lines if ln.lower() != pivot.lower()]
        text = pivot if len(uniq) == 0 else "\n".join([pivot] + uniq[:2])

    # Moderación
    if any(bw in text.lower() for bw in BAD_WORDS):
        text = ("Lamento el malentendido. Mantengamos una conversación cordial. "
                "¿Puedo ayudarte con información sobre productos, envíos o devoluciones?")
        log_event("moderation_replace", reason="bad_words")

    tox = get_tox()
    if tox is not None:
        try:
            scores = tox.predict(text)
            flags = [scores.get(k, 0) for k in (
                'toxicity','insult','obscene','threat','identity_attack','sexually_explicit'
            )]
            if any(s > 0.50 for s in flags):
                text = ("Prefiero mantener un tono cordial para ayudarte mejor. "
                        "¿Te apoyo con precios, disponibilidad o seguimiento de pedidos?")
                log_event("moderation_replace", reason="toxicity_threshold")
        except Exception as e:
            print(f"[WARN] Detoxify predict falló: {e}")

    # Refinado + memoria de idioma
    text = _refine_concise(text, lang)
    if len(text) < 2:
        text = "Con gusto te ayudo con productos, envíos y devoluciones. ¿Qué necesitas?"
    conv_state["last_lang"] = lang
    log_event("reply", lang=lang, text=text)
    return text, conv_state

# =====================================================
# [S11] Wrapper para Gradio (type="messages")
# =====================================================
@torch.inference_mode()
def generate_reply_messages(
    messages,
    pref_lang="auto",
    conv_state=None,
    *args,
    **kwargs
):
    """
    Devuelve (texto, conv_state). Usa conv_state['last_lang'] como memoria de idioma.
    Añade:
      - Truncado defensivo de entrada (caracteres y líneas)
      - Rate limit por sesión (ventana deslizante)
      - Mensajes de RL en el idioma del dropdown si está fijado (ES/EN), de lo contrario usa last_lang
    """
    # ---------- Configuración de seguridad ----------
    TRUNC_MAX_CHARS = int(os.getenv("TRUNC_MAX_CHARS", "1200"))   # máx. 1200 chars por turno
    TRUNC_MAX_LINES = int(os.getenv("TRUNC_MAX_LINES", "12"))     # máx. 12 líneas por turno
    RL_MAX_CALLS    = int(os.getenv("RL_MAX_CALLS", "8"))         # máx. 8 turnos por ventana
    RL_WINDOW_SECS  = int(os.getenv("RL_WINDOW_SECS", "60"))      # ventana 60s
    RL_MIN_INTERVAL = float(os.getenv("RL_MIN_INTERVAL", "1.0"))  # al menos 1.0s entre turnos

    conv_state = conv_state or {}

    # ---------- Normalización del historial ----------
    def normalize(mm):
        if isinstance(mm, str):
            return mm, []
        if not mm:
            return "", []
        if isinstance(mm, list) and all(isinstance(m, dict) for m in mm):
            user_msg = next((m.get("content", "") for m in reversed(mm) if m.get("role") == "user"), "")
            pairs, last = [], None
            for m in mm:
                r, c = m.get("role"), m.get("content", "")
                if r == "user":
                    last = c
                elif r == "assistant" and last is not None:
                    pairs.append((last, c)); last = None
            return user_msg, pairs
        if isinstance(mm, list) and all(isinstance(m, str) for m in mm):
            user_msg, prev = mm[-1], mm[:-1]
            pairs = [(prev[i], prev[i+1] if i+1 < len(prev) else "") for i in range(0, len(prev), 2)]
            return user_msg, pairs
        return (str(mm[-1]) if isinstance(mm, list) else str(mm)), []

    user_msg, history_pairs = normalize(messages)

    if len(history_pairs) > 6:
        history_pairs = history_pairs[-6:]

    # ---------- Truncado defensivo del input ----------
    if user_msg:
        lines = user_msg.splitlines()
        if len(lines) > TRUNC_MAX_LINES:
            lines = lines[:TRUNC_MAX_LINES]
        user_msg = "\n".join(lines)
        if len(user_msg) > TRUNC_MAX_CHARS:
            user_msg = user_msg[:TRUNC_MAX_CHARS]

    # ---------- Rate limiting por sesión ----------
    now = time.time()
    rl = conv_state.get("rl") or {"hits": []}   # lista de timestamps (segundos)
    hits = rl.get("hits", [])

    # purga fuera de ventana
    hits = [t for t in hits if now - t <= RL_WINDOW_SECS]
    rl["hits"] = hits

    # idioma de mensajes RL: prioriza dropdown, si no, last_lang (default 'es')
    rl_lang = {"es": "es", "en": "en"}.get(pref_lang, conv_state.get("last_lang") or "es")

    # anti-spam: intervalo mínimo
    if hits and (now - hits[-1] < RL_MIN_INTERVAL):
        msg = "Por favor espera un instante antes de enviar otro mensaje." \
              if rl_lang == "es" else "Please wait a moment before sending another message."
        log_event("rate_limit_min_interval", last_dt=now - hits[-1])
        conv_state["rl"] = rl
        return msg, conv_state

    # límite de llamadas por ventana
    if len(hits) >= RL_MAX_CALLS:
        msg = (f"Has alcanzado el límite de {RL_MAX_CALLS} mensajes por {RL_WINDOW_SECS}s. "
               f"Intenta de nuevo en breve.") if rl_lang == "es" \
              else (f"You have reached the limit of {RL_MAX_CALLS} messages per {RL_WINDOW_SECS}s. "
                    f"Please try again shortly.")
        log_event("rate_limit_window_exceeded", hits=len(hits))
        conv_state["rl"] = rl
        return msg, conv_state

    # registra este intento
    hits.append(now)
    rl["hits"] = hits
    conv_state["rl"] = rl

    # ---------- Resolución de idioma (memoria + auto) ----------
    short = len((user_msg or "").strip()) < 3
    if pref_lang == "auto" and short and conv_state.get("last_lang"):
        lang = conv_state["last_lang"]
    else:
        lang = resolve_lang(user_msg, pref_lang, history_pairs)

    log_event("turn", pref=pref_lang, resolved=lang, user=user_msg[:120])  # log truncado

    # ---------- Llamada al motor principal ----------
    reply_text, conv_state = generate_reply(
        user_msg, history_pairs, lang, pref_lang, conv_state=conv_state
    )
    return reply_text, conv_state

# ============================================
# [S12] Quick Eval (herramienta dev)
# ============================================
def quick_eval(pref_lang):
    tests = [
        ("es", "¿Hacen envíos internacionales?"),
        ("es", "Quiero devolver un producto defectuoso, orden #A1B2C3."),
        ("es", "¿Tienen stock de laptops gamer?"),
        ("en", "Do you ship to Canada?"),
        ("en", "I need to return an item, order #ZX-991-OK."),
        ("en", "Is a 16-inch ultralight available under $1200?"),
    ]
    rows = []
    for lang_hint, q in tests:
        t0 = time.perf_counter()
        reply_text, _ = generate_reply(q, [], lang_hint, lang_hint)  # <- desempaquetado
        ms = int((time.perf_counter() - t0) * 1000)
        rows.append((lang_hint, q, reply_text, ms))
        log_event("quick_eval_item", lang=lang_hint, ms=ms, q=q, reply=reply_text)

    md = ["| Lang | Prompt | ms | Reply (trunc) |",
          "|---|---|---:|---|"]
    for lang_hint, q, r, ms in rows:
        trunc = (r[:80] + "…") if len(r) > 80 else r
        md.append(f"| {lang_hint} | {q.replace('|','/')} | {ms} | {trunc.replace('|','/')} |")
    return "\n".join(md)


# ============================================
# [S13] Interfaz de usuario Gradio (Blocks UI)
# ============================================
TITLE = "TechStore · Asistente Virtual"
DESC = "Chatbot de atención al cliente bilingüe (ES/EN). Prueba: envíos, devoluciones, stock, horarios."

CSS = """
#header { display:flex; align-items:center; gap:.75rem; margin-bottom:.5rem; }
#brand { font-weight:700; font-size:1.1rem; }
.gradio-container { max-width: 980px !important; }
.quick > button { width:100%; }
"""

HEADER_HTML = """
<div id="header">
  <img src="assets/logo.png" style="height:28px;border-radius:8px"/>
  <div id="brand">TechStore · Asistente Virtual</div>
</div>
"""

theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

def add_user_msg(messages, user_text):
    """Agrega turno del usuario y limpia textbox."""
    if messages is None:
        messages = []
    messages = list(messages) + [{"role": "user", "content": user_text}]
    return messages, gr.update(value="")

def bot_reply(messages, pref_lang, conv_st):
    """Llama al motor y agrega turno del asistente. Retorna (state, chatbot, conv_state)."""
    reply_text, conv_st = generate_reply_messages(messages, pref_lang, conv_st)
    if isinstance(reply_text, dict):
        reply_text = reply_text.get("content", "")
    if not isinstance(reply_text, str):
        reply_text = str(reply_text)
    messages = list(messages) + [{"role": "assistant", "content": reply_text}]
    return messages, messages, conv_st

def quick_inject(messages, pref_lang, conv_st, text):
    """Inyecta un atajo como usuario y responde."""
    if messages is None:
        messages = []
    messages = list(messages) + [{"role": "user", "content": text}]
    return bot_reply(messages, pref_lang, conv_st)

with gr.Blocks(theme=theme, css=CSS) as demo:
    gr.HTML(HEADER_HTML)
    gr.Markdown(DESC)

    with gr.Row():
        # ---------- Columna principal (chat) ----------
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                type="messages",
                height=520,
                show_copy_button=True,
                avatar_images=("assets/user.png", "assets/bot.png")
            )
            txt = gr.Textbox(placeholder="Escribe tu consulta… (ES/EN)", autofocus=True)
            clear = gr.Button("Limpiar chat")

            # Botón catálogo de ejemplo (opcional)
            btn_catalog = gr.Button("🧾 Laptops <$1200")

            # Botón 'Nueva consulta' (reinicia la conversación pero conserva idioma/memoria)
            btn_new_conv = gr.Button("🆕 Nueva consulta")

        # ---------- Columna lateral (atajos y dev) ----------
        with gr.Column(scale=1):
            gr.Markdown("### Atajos")
            lang_dd = gr.Dropdown(choices=["Auto", "Español", "English"], value="Auto", label="Idioma")

            with gr.Column(elem_classes=["quick"]):
                b1 = gr.Button("📦 Envíos")
                b2 = gr.Button("↩️ Devoluciones")
                b3 = gr.Button("🧾 Stock")

            gr.Markdown("### Dev")
            eval_btn = gr.Button("▶️ Quick Eval")
            eval_out = gr.Markdown(value="(sin ejecutar)")

    # ---------- Estados ----------
    messages_state = gr.State([])
    language_state = gr.State("auto")
    conv_state = gr.State({})  # memoria ligera (p. ej. último idioma)

    # Mapear dropdown -> estado interno
    def _map_lang(v):
        return {"Auto": "auto", "Español": "es", "English": "en"}.get(v, "auto")
    lang_dd.change(_map_lang, lang_dd, language_state)

    # Enviar texto
    txt.submit(
        add_user_msg, [messages_state, txt], [messages_state, txt]
    ).then(
        bot_reply, [messages_state, language_state, conv_state],
        [messages_state, chatbot, conv_state]
    )

    # Limpiar (borra historial y memoria de conv)
    clear.click(lambda: ([], [], {}), None, [messages_state, chatbot, conv_state])

    # 'Nueva consulta' (reinicia solo la conversación, conserva conv_state)
    btn_new_conv.click(lambda m, cs: ([], []), inputs=[messages_state, conv_state], outputs=[messages_state, chatbot])

    # Atajos (pasan idioma preferido + conv_state)
    b1.click(
        quick_inject,
        [messages_state, language_state, conv_state, gr.State("¿Ofrecen envíos internacionales?")],
        [messages_state, chatbot, conv_state]
    )
    b2.click(
        quick_inject,
        [messages_state, language_state, conv_state, gr.State("Quiero hacer una devolución")],
        [messages_state, chatbot, conv_state]
    )
    b3.click(
        quick_inject,
        [messages_state, language_state, conv_state, gr.State("¿Tienen stock de laptops?")],
        [messages_state, chatbot, conv_state]
    )

    # Catálogo: ejemplo de acción predefinida (usa S7.action_laptops_under)
    # Si tu action_laptops_under devuelve (messages, chatbot), está OK:
    btn_catalog.click(
        action_laptops_under,
        inputs=[messages_state, language_state],
        outputs=[messages_state]
    ).then(
        bot_reply,
        inputs=[messages_state, language_state, conv_state],   
        outputs=[messages_state, chatbot, conv_state]          
    )

    # Quick Eval
    eval_btn.click(lambda pref: quick_eval(pref), inputs=[language_state], outputs=[eval_out])

if __name__ == "__main__":
    auth_pair = None
    u = os.getenv("APP_USER"); p = os.getenv("APP_PASS")
    if u and p:
        auth_pair = (u, p)
    demo.title = TITLE
    demo.launch(
        share=False,
        auth=auth_pair,
        server_name=os.getenv("SERVER_HOST", "127.0.0.1"),
        server_port=int(os.getenv("SERVER_PORT", "7860")),
    )