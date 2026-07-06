# 🚀 Qwen Autopilot Support Engine

An autonomous, AI-powered customer support system featuring a dynamic Human-in-the-Loop (HITL) Manager Dashboard. Built for the **Global AI Hackathon**.

## 📖 Overview

The Qwen Autopilot Support Engine is a decoupled, full-stack application designed to handle frontline customer support tickets autonomously while seamlessly escalating complex or high-risk issues to human managers. 

By leveraging the **Qwen LLM** via **LangGraph**, the system acts as an intelligent routing engine. It can dynamically request missing information (like order IDs), enforce policy requirements (like demanding photo uploads for damaged goods), and maintain a live, bidirectional chat session with the customer throughout the entire resolution process.

## ✨ Key Features

* **🧠 Autonomous Intent & Fraud Detection:** Evaluates incoming tickets for sentiment, intent, and prompt-injection/fraud risks.
* **📸 Visual Proof Enforcement:** Automatically halts the resolution process to request a photo upload if a customer claims an item is broken or a package is empty.
* **⏸️ Sticky Human-in-the-Loop Escalations:** High-risk tickets or items with attachments are pushed to a live Manager Queue. The AI state "freezes" so subsequent casual chat messages don't overwrite the original intent snapshot.
* **💬 Bidirectional Live Chat:** Customers stay in an active chat session (persisted via `localStorage`) while managers review their case, rather than hitting a static "Please wait for an email" wall.
* **🎛️ Master-Detail Manager Dashboard:** A sleek, dark-mode command center where managers can monitor a real-time queue, review full chat transcripts, view Base64 image attachments inline, and execute final approvals/rejections.

## 🛠️ Tech Stack

* **AI & Logic:** Python, LangGraph, Qwen LLM, OpenAI SDK
* **Backend:** FastAPI, Pydantic (Strict Schema Validation)
* **Frontend:** HTML5, Vanilla JavaScript, Tailwind CSS (CSS Grid layouts)

## 🚀 Quick Start Guide

### Prerequisites
* Python 3.8+
* An active API key compatible with the Qwen LLM endpoint.

### 1. Clone the repository
```bash
git clone [https://github.com/AMAN0BLAZY/qwen-autopilot-engine.git](https://github.com/AMAN0BLAZY/qwen-autopilot-engine.git)
cd qwen-autopilot-engine
```

### 2. Install Dependencies
```bash
pip install fastapi uvicorn langgraph langchain-openai python-dotenv pydantic
```

### 3. Environment Variables
Create a .env file in the root directory and add your API key:
Code snippet:
```bash
OPENAI_API_KEY=your_qwen_api_key_here
OPENAI_API_BASE=your_qwen_endpoint_url_here
```

### 4. Run the Backend Server
Start the FastAPI server:
```bash
python app.py
"The server will run locally on http://127.0.0.1:8000."
```

### 5. Launch the Application
Simply double-click the index.html file to open it in your web browser.
The app runs as a dual-pane interface so you can simulate both the Customer and Manager experiences simultaneously.

💡 Architecture Notes
This repository represents a hackathon proof-of-concept. In a production environment, the architecture would be decoupled further:

The Customer View would be injected as an embeddable chat widget.

The Manager View would live behind enterprise SSO.

The in-memory MemorySaver and queues would be migrated to PostgreSQL or Redis to ensure zero state loss during server restarts.

Outbound notifications would be handled by a transactional SMTP provider (e.g., SendGrid).

