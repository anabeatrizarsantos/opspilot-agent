# 🤖 OpsPilot

**OpsPilot** is an LLM-powered operations support agent designed to understand user requests, classify intentions, and automate operational workflows.

The project focuses on building a production-style AI agent architecture rather than a simple chatbot.

---

## 🎯 Goal

OpsPilot aims to simulate a real internal support automation system capable of:

* understanding natural language requests
* routing tasks to the correct action
* generating structured operational tickets
* retrieving knowledge base information
* assisting users in operational processes

This repository documents the incremental development of an AI agent system using Azure OpenAI.

---

## 🧠 Current Capabilities (v0.1)

At this stage the system can:

* connect securely to Azure OpenAI
* send and receive messages from a deployed LLM
* apply a system prompt (agent identity)
* run a local CLI interface for testing

The project currently implements the **LLM Connectivity Layer**, the foundation of any production AI agent.

---

## 🏗️ Architecture (current)

```
User (CLI)
     ↓
main.py
     ↓
llm_client.py  ← LLM Abstraction Layer
     ↓
Azure OpenAI (o3-mini deployment)
```

The LLM is never accessed directly by other modules.
All future components (router, tools, memory, RAG) will interact through the `ask_llm()` interface.

---

## 🧩 Tech Stack

* Python 3.11
* Azure OpenAI
* OpenAI Python SDK
* dotenv
* uv (environment & dependency management)

---

## 📖 Purpose of the Project

This project is part of a hands-on study of **AI Engineering and LLM agent systems**.

The goal is not only to use LLMs, but to design a modular architecture similar to real production AI systems:

* abstraction layers
* routing logic
* structured outputs
* tool integration

---

## 👤 Author

Ana Santos
AI Engineering • LLM Systems • Backend Integration
