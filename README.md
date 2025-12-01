# 🤖 AI Chatbot Assistant

> Conversational AI system with natural language understanding and secure API deployment

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

[🚀 Quick Start](#quick-start) • [🔐 Security](#security-features) • [📚 API Docs](#api-documentation)

---

## 📋 Overview

Production-ready chatbot API built with FastAPI, featuring:
- Natural language processing for intent recognition
- RESTful API with automatic documentation
- Full CI/CD pipeline with GitHub Actions
- DevSecOps best practices (CodeQL, Secret Scanning)
- Docker containerization for easy deployment

---

## 🛠️ Tech Stack

- **Backend:** FastAPI (async Python web framework)
- **NLP:** [Specify: OpenAI API / Hugging Face / Custom model]
- **Security:** CodeQL static analysis, dependency scanning
- **CI/CD:** GitHub Actions
- **Containerization:** Docker
- **Testing:** Pytest

---

## 🚀 Quick Start

### Run with Docker (Recommended)
```bash
# Clone repository
git clone https://github.com/Geerazo/chatbot-ai.git
cd chatbot-ai

# Build and run
docker-compose up -d

# API available at http://localhost:8000
```

### Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export API_KEY=your_api_key_here

# Start server
uvicorn main:app --reload
```

### Test the API
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how can you help me?"}'
```

---

## 📚 API Documentation

### Endpoints

#### `POST /chat`
Send a message to the chatbot

**Request:**
```json
{
  "message": "What's the weather like?",
  "user_id": "user123",
  "session_id": "session456"
}
```

**Response:**
```json
{
  "response": "I can help you check the weather...",
  "intent": "weather_query",
  "confidence": 0.95,
  "timestamp": "2025-11-27T10:30:00Z"
}
```

#### `GET /health`
Check API health status

**Interactive docs:** http://localhost:8000/docs

---

## 🔐 Security Features

### Implemented DevSecOps Practices

✅ **CodeQL Analysis**  
- Automated code scanning for vulnerabilities
- Runs on every pull request

✅ **Dependency Scanning**  
- GitHub Dependabot alerts
- Automatic security updates

✅ **Secret Detection**  
- Pre-commit hooks to prevent API key leaks
- GitHub Secret Scanning enabled

✅ **Input Validation**  
- Pydantic schemas for request validation
- Protection against injection attacks
- Rate limiting (100 req/min per user)

✅ **ReDoS Prevention**  
- Optimized regular expressions
- Timeout limits for pattern matching

---

## 🏗️ Architecture
```
User Request
    ↓
FastAPI Server
    ↓
Input Validation (Pydantic)
    ↓
NLP Processing
    ├── Intent Recognition
    ├── Entity Extraction
    └── Context Management
    ↓
Response Generation
    ↓
JSON Response
```

---

## 🧪 Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/test_chat.py::test_basic_conversation
```

---

## 📦 Deployment

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### GitHub Actions CI/CD
```yaml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest
      
  security:
    runs-on: ubuntu-latest
    steps:
      - name: CodeQL Analysis
        uses: github/codeql-action/analyze@v2
```

---

## 🔮 Future Enhancements

- [ ] Multi-language support (ES, EN, PT)
- [ ] Voice input/output integration
- [ ] Conversational memory (Redis)
- [ ] Admin dashboard for analytics
- [ ] Webhooks for third-party integrations

---

## 📝 License

MIT License - see [LICENSE](LICENSE)

---

## 📫 Contact

**Edgar Erazo**  
📧 eerazo83@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/edgar-andres-erazo)  
💻 [GitHub](https://github.com/Geerazo)

---

⭐️ **Star this repo if you find it useful!**
```