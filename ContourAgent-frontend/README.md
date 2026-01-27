# ContourAgent-frontend-java

## Overview

This repository provides the frontend component of the ContourAgent framework.
The frontend is responsible for user interaction, task specification, and visualization of geological contour mapping results, serving as the interface between users and the backend geospatial modeling services.

The frontend is designed to support interactive exploration and visual inspection of contour maps generated through automated geostatistical workflows.

## Scientific Purpose

In geological and paleogeographical studies, contour maps play a critical role in representing spatial patterns and subsurface structures. While backend interpolation algorithms ensure numerical accuracy, interactive visualization and human-in-the-loop inspection remain essential for interpretation and validation.

This frontend application aims to:

- Facilitate natural-language-driven task submission,

- Provide intuitive visualization of contour generation results,

- Support reproducible interpretation of geospatial analysis outcomes.

## Methodology Correspondence

The frontend implementation corresponds to the methodological components described in the associated manuscript as follows:

- Natural language task specification and interaction: Section 2.1

- Multi-agent collaboration and task execution feedback: Section 2.2

- Contour map generation result visualization: Section 2.3

The frontend does not perform geostatistical modeling directly; instead, it visualizes and manages results produced by the backend services.

## Software Architecture

The frontend is implemented as a client-side web application and communicates with the backend via RESTful APIs using JSON-formatted messages.

Its primary responsibilities include:

- User interaction and task input,

- Request dispatching to backend services,

- Visualization of contour maps and associated geospatial data.

## Technologies and Libraries

The frontend is developed using the following technologies:

- Vue.js 3__: Progressive JavaScript framework for building user interfaces.

- Vite: Frontend build and development tooling

- Vue Router: Client-side routing

- Vuex: State management

- Element Plus: UI component library

- OpenLayers: Interactive geospatial map rendering

- D3.js: Data-driven contour visualization

- ECharts / Plotly.js: Auxiliary charting and data visualization

- Turf.js: Client-side geospatial operations

- Axios: HTTP communication with backend services

- These libraries collectively support interactive mapping, visualization, and data exploration.

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
Requirements

Node.js 22 or later

npm or equivalent package manager

Installation

```text
npm install
```


## Usage and Reproducibility

To start the frontend locally:

```text
npm run dev
```

The application will run locally (default: http://localhost:3003) and connect to the backend service for task execution and result visualization.


## Limitations

The frontend focuses on visualization and interaction and does not perform numerical geostatistical computations.
Advanced uncertainty visualization and real-time collaborative editing are not included in the current implementation.

## License

This project is released under the MIT License.
