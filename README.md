# ContourAgent

## Overview

ContourAgent is a natural-language-driven intelligent framework for geological contour map generation. The system integrates large language models (LLMs), a multi-agent architecture, and geostatistical interpolation techniques to support automated and reproducible contour mapping workflows.

## Scientific Background and Purpose

Geological contour map generation is a fundamental task in spatial analysis and geoscience research. Conventional workflows require extensive manual parameter configuration and expert knowledge, limiting efficiency and reproducibility. This software aims to reduce manual intervention by enabling natural language–based task specification and automated spatial interpolation.

## Methodology Correspondence

The software implementation corresponds directly to the methodology described in the associated manuscript:

- **Natural language parsing and task decomposition:** Section 2.1
- **Multi-agent orchestration (based on MCP):** Section 2.2
- **Contour map generation and visualization:** Section 2.3

## Software Architecture

The framework adopts a decoupled Java–Python architecture:

- **Java frontend (`ContourAgent-frontend`):** Responsible for user interaction, task submission, and quick testing.
- **Python backend (`ContourAgent-backend`):** Responsible for agent reasoning, MCP-based context management, geostatistical modeling, and contour map generation.

The two components communicate via RESTful APIs using JSON-formatted messages.

## Repository Structure

The project is organized into two main directories:

- `ContourAgent-backend/`: Contains the Python source code, API server (`api.py`), and dependencies.
- `ContourAgent-frontend/`: Contains the Java Spring Boot application and a lightweight testing interface (`test-quick.html`).

## Requirements and Dependencies

### Python Backend
- Python 3.11
- All dependencies listed in `ContourAgent-backend/requirements.txt` (e.g., `numpy`, `scipy`, `pykrige`, `shapely`, etc.)

### Java Frontend
- Java 11 or later
- Maven

## Usage and Reproducibility

### 1. Setup and Start Backend (Python)

Navigate to the backend directory, install dependencies, and start the API server:

```bash
cd ContourAgent-backend
pip install -r requirements.txt
python api.py

### 2. Frontend Quick Test

A quick test file test-quick.html has been added to the ContourAgent-frontend directory. If you wish to test the interface rapidly without launching the full Spring Boot application, run the following command:

```bash
cd ContourAgent-frontend
python start_http_server.py


### 3. Start Java Frontend (Standard Mode)

To start the frontend locally:

```bash
npm run dev