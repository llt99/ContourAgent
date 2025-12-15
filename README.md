# ContourAgent

## Overview

ContourAgent is a natural-language-driven intelligent framework for geological contour map generation. The system integrates large language models (LLMs), a multi-agent architecture, and geostatistical interpolation techniques to support automated and reproducible contour mapping workflows.

## Scientific Background and Purpose

Geological contour map generation is a fundamental task in spatial analysis and geoscience research. Conventional workflows require extensive manual parameter configuration and expert knowledge, limiting efficiency and reproducibility. This software aims to reduce manual intervention by enabling natural language–based task specification and automated spatial interpolation.

## Methodology Correspondence

The software implementation corresponds directly to the methodology described in the associated manuscript:

- Natural language parsing and task decomposition: Section 2.1
- Multi-agent orchestration (based on MCP): Section 2.2
- Contour map generation and visualization: Section 2.3

## Software Architecture

The framework adopts a decoupled Java–Python architecture:

- Java frontend: responsible for user interaction and task submission
- Python backend: responsible for agent reasoning, MCP-based context management, geostatistical modeling, and contour map generation

The two components communicate via RESTful APIs using JSON-formatted messages.

## Repository Structure

The project consists of the following repositories:

- Java frontend: https://github.com/llt99/contouragent-frontend-java
- Python backend: https://github.com/llt99/contouragent-backend-python

## Requirements and Dependencies

### Python Backend
- Python 3.8 or later
- All dependencies listed in `requirements.txt` (e.g., `numpy`, `scipy`, `pykrige`, `shapely`, etc.)


### Java Frontend
- Java 17 or later
- Maven


## Usage and Reproducibility

1. Install Python dependencies:
   pip install -r requirements.txt

2. Start the Python backend:
   python main.py

3. Start the Java frontend:
   mvn spring-boot:run

The backend service runs locally and processes geological contour generation tasks submitted from the frontend.
