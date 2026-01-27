# ContourAgent

**A Natural-Language-Driven Intelligent Framework for Automated Geological Contour Map Generation**

**ContourAgent** integrates **Large Language Models (LLMs)**, a **Multi-Agent Collaboration Architecture**, and **Geostatistical Interpolation Methods** to support reproducible and low-intervention geological mapping workflows.

---

##  1. Background and Motivation

Geological contour mapping is a core task in geoscience research and spatial analysis. Traditional workflows rely heavily on manual parameter configuration and expert knowledge, often resulting in **low efficiency** and **limited reproducibility**.

**ContourAgent** addresses these limitations by enabling **natural language–driven mapping tasks**. It allows users to describe geological mapping requirements in domain language while the system automatically performs:
* Task parsing
* Agent coordination
* Spatial interpolation

---

##  2. Methodological Correspondence

This repository provides the software implementation corresponding directly to the methodology described in the associated manuscript. This correspondence ensures methodological transparency and experimental reproducibility.

| Software Module | Manuscript Section |
|-----------------|--------------------|
| Natural language understanding and task parsing | **Section 2.1** |
| MCP-based multi-agent orchestration and context management | **Section 2.2** |
| Geostatistical interpolation and contour visualization | **Section 2.3** |

---

##  3. System Architecture

ContourAgent adopts a **decoupled frontend–backend architecture** implemented using Node.js and Python.

### 3.1 Frontend
* **Module:** `ContourAgent-frontend`
* **Technology:** Node.js, npm
* **Responsibilities:**
    * User interaction and task submission.
    * Natural language input testing.
    * Frontend visualization and API request dispatch.

### 3.2 Backend
* **Module:** `ContourAgent-backend`
* **Technology:** Python
* **Responsibilities:**
    * LLM-driven intent recognition and semantic parsing.
    * Multi-agent task scheduling based on **MCP (Model Context Protocol)**.
    * Context memory and execution state management.
    * Geostatistical interpolation (e.g., IDW, Kriging).
    * Contour generation and spatial data output.

### 3.3 Communication
Frontend and backend communicate via **RESTful APIs**, exchanging **JSON-formatted structured task instructions**.

---

##  4. Repository Structure

```text
ContourAgent/
├── ContourAgent-backend/
│   ├── api.py                 # Backend API entry
│   ├── requirements.txt       # Python dependencies
│   └── ...                    # Agent logic, interpolation, visualization
│
├── ContourAgent-frontend/
│   ├── test-quick.html        # Lightweight frontend testing interface
│   ├── start_http_server.py   # Simple HTTP server for quick testing
│   └── ...                    # Frontend source code (Node.js)
```

---

##  5. Requirements

### 5.1 Backend (Python)
* **Python 3.11**
* Key dependencies (`requirements.txt`):
    * `numpy`
    * `scipy`
    * `pykrige`
    * `shapely`
    * `matplotlib`

### 5.2 Frontend (Node.js)
* **Node.js 18+**
* npm

---

##  6. Usage Guide

### 6.1 Start Backend (Python)
The backend API server must be running to process requests.

```bash
cd ContourAgent-backend
pip install -r requirements.txt
python api.py
```

### 6.2 Frontend Quick Test (HTML)
For rapid testing without launching the full frontend development environment, a lightweight HTML-based interface is provided.

```bash
cd ContourAgent-frontend
python start_http_server.py
```
> **Note:** Then open `test-quick.html` via the local HTTP server address printed in the console.
>
> **Warning:** Do not open the HTML file directly using the `file://` protocol to avoid CORS issues.

### 6.3 Frontend Development Mode (npm)
For full frontend features:

```bash
cd ContourAgent-frontend
npm install
npm run dev
```
>  **Ensure the backend (`api.py`) is running before starting the frontend.**

---

##  7. Reproducibility and Extensibility

* **Traceability:** All task parsing results, agent decisions, and interpolation parameters are recorded in structured form.
* **Extensibility:** The modular multi-agent design allows easy extension, including:
    * Adding new interpolation methods.
    * Integrating additional geological rules.
    * Supporting multi-factor or facies mapping tasks.

---

##  8. Intended Use

ContourAgent is designed for:
* Geological and sedimentological research.
* Intelligent geological mapping experiments.
* Validation of natural language–driven scientific workflows.

*The system is intended to assist rather than replace geological expertise, providing a reproducible and extensible computational framework for expert knowledge formalization.*

## 9. License

This project is released under the MIT License.
