# Multi-Agent Travel Planner AI

A **modular, multi-agent travel planning system** built with LangChain, LangGraph, and Google Gemini LLM. This project generates structured travel plans including **hotels, transport, activities, events, weather forecasts, food recommendations, and budget breakdowns**.

The system uses a **supervised multi-agent architecture**: each agent handles a distinct part of the planning process, and a **supervisor agent** orchestrates the workflow to ensure completion and consistency.

---

## Features

- **Context & Budget Agent**: Fetches weather forecasts, upcoming events, and calculates a realistic budget.  
- **Schedule & Activity Agent**: Suggests 6–9 activities per destination and distributes them across daily time slots.  
- **Logistics Agent**: Recommends hotels, transport, and food options with pricing and ratings.  
- **Supervisor Agent**: Ensures all tasks are completed and generates a final structured travel plan.  
- **Completion Tracking**: Prevents unnecessary iterations and infinite loops.  
- **Real-Time Data Integration**: Uses APIs like Geoapify, OpenWeather, and Ticketmaster for accurate recommendations.

---

## Tech Stack

- **Language**: Python  
- **LLM**: Google Gemini (`langchain_google_genai`)  
- **Agent Orchestration**: LangGraph (`StateGraph`, `MessagesState`)  
- **APIs**: Geoapify, OpenWeather, Ticketmaster  
- **Data Handling**: Dataclasses for structured travel information

---
