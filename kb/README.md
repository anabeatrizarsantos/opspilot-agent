# Aurora Beauty Studio — Knowledge Base

This folder contains the source documents used by OpsPilot's Retrieval-Augmented Generation (RAG) pipeline.

All documents are written in plain text (.txt) and represent structured business knowledge for the fictional client "Aurora Beauty Studio".

## Structure

kb/
  services/
    services.txt
    pricing.txt

  policies/
    booking_policy.txt

  studio/
  about.txt
  hours_location.txt
  contact.txt

## Purpose

- Each file represents a logical knowledge domain.
- During indexing, documents are chunked into smaller segments.
- Each chunk is converted into a vector embedding.
- The vector index is stored under `data/kb/` (ignored by git).

## Design Principles

- Clear separation of business domains.
- Human-readable source documents.
- Scalable structure for future expansion.
- Optimized for semantic retrieval (RAG).

This knowledge base enables the AI assistant to provide grounded, context-aware responses instead of generating generic answers.