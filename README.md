# Forsa Smart Chatbot

An intelligent chatbot system for Algérie Télécom, powered by **local AI** (Qwen 2.5 3B) and RAG (Retrieval-Augmented Generation) for accurate document-based responses.

##  Overview

This chatbot provides instant answers about:
-  **Guides** - Internal procedures and processes
-  **Conventions** - Partnership agreements
-  **Produits** - Products and equipment
-  **Offres** - Commercial offers and pricing

## 📸 Screenshots

### Landing Page
![Landing Page](assets/landing-page.png)

### Chat Interface
![Chat Interface](assets/chat-interface2.png)
![Chat Interface](assets/chat-interface.png)


### Chat History
![Chat History](assets/historique-page.png)

### Document Library
![Document Library](assets/document-library.png)

##  Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Services                      │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │  MinIO   │  │ Retrieval    │  │   Frontend      │    │
│  │  S3      │  │ API          │  │   React + Vite  │    │
│  │  :9010   │  │ :8000        │  │   :5173         │    │ 
│  └──────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────┘
                         ↓ ↑
┌─────────────────────────────────────────────────────────┐
│              Host Machine (with GPU)                    │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Chatbot API (FastAPI + Qwen 2.5 3B)              │  │
│  │  • RAG Pipelines (Guides, Offers, etc.)           │  │
│  │  • Local LLM Inference with GPU                   │  │
│  │  • Port: 8001                                     │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

##  Quick Start

### Prerequisites

- **Docker & Docker Compose**
- **NVIDIA GPU** with CUDA support (6GB+ VRAM recommended)
- **Conda** or Python 3.10+ environment
- **Node.js** (for frontend development)


### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/sohaib162/forsa-smart-chatbot
cd forsa-smart-chatbot
```


#### 2. Start Docker Services

```bash
# Start MinIO, Retrieval API, and Frontend
sudo docker compose up -d

# Check services are running
sudo docker compose ps
```

This starts:
- **MinIO** (S3 storage) - http://localhost:9011
- **Retrieval API** - http://localhost:8000
- **Frontend** - http://localhost:5173

#### 3. Setup Python Environment for Chatbot API


**Create new conda environment**

```bash
# Create environment
conda create -n forsa-chatbot python=3.10 -y
conda activate forsa-chatbot

# Install dependencies
cd chat-bot-algerie-telecom
pip install -r requirements.txt
```


#### 4. Start the Chatbot API with GPU

```bash
# From project root
./run-chatbot-local.sh
```

#### 5. Access the Application

- **Frontend UI**: http://localhost:5173
- **Chatbot API**: http://localhost:8001
- **Retrieval API**: http://localhost:8000
- **MinIO Console**: http://localhost:9011 (minioadmin/minioadmin)



##  Development

### Running Frontend in Development Mode

```bash
cd forsa-frontend
npm install
npm run dev
```




##  Project Structure

```
forsa-smart-chatbot/
├── chat-bot-algerie-telecom/      # Chatbot API (FastAPI)
│   ├── pipelines/                 # RAG pipelines by category
│   │   ├── offers/               # Commercial offers
│   │   ├── guide/                # Internal guides
│   │   ├── conventions/          # Conventions
│   │   └── depot/                # Products
│   ├── local_llm_client.py       # Local Qwen LLM interface
│   ├── main.py                   # FastAPI application
│   └── requirements.txt
├── forsa-endpoints/              # Retrieval API
│   ├── S3_Storage/              # S3 integration
│   └── main.py
├── forsa-frontend/              # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── MarkdownMessage.tsx
│   │   │   └── TypingMarkdownMessage.tsx
│   │   └── lib/
│   └── package.json
├── docker-compose.yml           # Docker services
├── run-chatbot-local.sh        # Start chatbot with GPU
└── README.md
```



