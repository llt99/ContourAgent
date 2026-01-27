# ContourAgent-frontend

## Overview

This repository provides the **frontend component** of the ContourAgent framework.
The frontend is responsible for user interaction, task specification, and visualization of geological contour mapping results, serving as the interface between users and the backend geospatial modeling services.

It is designed to support **interactive exploration** and **visual inspection** of contour maps generated through automated geostatistical workflows.

## Scientific Purpose

In geological and paleogeographical studies, contour maps play a critical role in representing spatial patterns and subsurface structures. While backend interpolation algorithms ensure numerical accuracy, interactive visualization and human-in-the-loop inspection remain essential for interpretation and validation.

This frontend application aims to:
- Facilitate **natural-language-driven task submission**.
- Provide **intuitive visualization** of contour generation results.
- Support **reproducible interpretation** of geospatial analysis outcomes.

##  Data Privacy and Test Datasets

**Note on Data Confidentiality:**
Due to strict confidentiality agreements and intellectual property protections regarding the geological exploration data used in the associated manuscript, the original raw field datasets are **not included** in this repository.

**Quick Test with Synthetic Data:**
To facilitate reproducibility and code verification, the provided **quick test files** and default development environment are configured to use **synthetic test data (desensitized mock data)**.
* These datasets mimic the structure and format of real geological data.
* They allow users to validate the visualization rendering and interaction logic immediately without requiring access to private sensitive data.

## Methodology Correspondence

The frontend implementation corresponds to the methodological components described in the associated manuscript as follows:

| Component | Manuscript Section |
|-----------|--------------------|
| Natural language task specification and interaction | **Section 2.1** |
| Multi-agent collaboration and task execution feedback | **Section 2.2** |
| Contour map generation result visualization | **Section 2.3** |

*The frontend does not perform geostatistical modeling directly; instead, it visualizes and manages results produced by the backend services.*

## Software Architecture

The frontend is implemented as a client-side web application and communicates with the backend via RESTful APIs using JSON-formatted messages.

Its primary responsibilities include:
- User interaction and task input.
- Request dispatching to backend services.
- Visualization of contour maps and associated geospatial data.

## Technologies and Libraries

The frontend is developed using the following technologies:

* **Vue.js 3**: Progressive JavaScript framework for building user interfaces.
* **Vite**: Frontend build and development tooling.
* **Vue Router**: Client-side routing.
* **Vuex**: State management.
* **Element Plus**: UI component library.
* **OpenLayers**: Interactive geospatial map rendering.
* **D3.js**: Data-driven contour visualization.
* **ECharts / Plotly.js**: Auxiliary charting and data visualization.
* **Turf.js**: Client-side geospatial operations.
* **Axios**: HTTP communication with backend services.

These libraries collectively support interactive mapping, visualization, and data exploration.

## Repository Structure

```text
.
├── public/                 # Static assets (e.g., GeoJSON, map tiles)
├── src/
│   ├── assets/             # Images and global styles
│   ├── components/         # Reusable Vue components
│   ├── router/             # Routing configuration
│   ├── store/              # State management
│   ├── views/              # Page-level components (e.g., AgentMapping.vue)
│   ├── App.vue             # Root component
│   └── main.js             # Application entry point
├── index.html
├── package.json
└── vite.config.js
```

## Requirements and Installation

### Requirements
* **Node.js**: 22 or later
* **npm** or equivalent package manager

### Installation

```bash
npm install
```

## Usage and Reproducibility

### 1. Frontend Quick Test (HTML)
For rapid testing without launching the full frontend development environment (Node.js/Vite), a lightweight HTML-based interface is provided.

**Note on Test Data:**
This quick test interface is pre-configured to use **synthetic/desensitized test data**. It allows for immediate visualization of the contour generation capabilities without requiring a live backend connection or access to sensitive raw datasets.

To launch the quick test:

```bash
python start_http_server.py
```

> **Note:** Then open `test-quick.html` via the local HTTP server address printed in the console (usually `http://localhost:8000/test-quick.html`).
>
> **Warning:** Do not open the HTML file directly using the `file://` protocol (i.e., double-clicking the file), as this will cause CORS errors and prevent the test data from loading.

### 2. Development Mode (Full Features)
To start the full Vue.js development environment:

```bash
npm run dev
```

The application will run locally (default: `http://localhost:3003`) and attempt to connect to the backend service.

> **Note:** If the backend is not running, the system may default to using embedded **synthetic test data** to demonstrate UI functionality.

## Limitations

* The frontend focuses on **visualization and interaction** and does not perform numerical geostatistical computations.
* Advanced uncertainty visualization and real-time collaborative editing are not included in the current implementation.

## License

This project is released under the **MIT License**.
