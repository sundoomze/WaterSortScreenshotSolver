# Water Sort Screenshot Solver

[Live on Github Pages](https://doublemover.github.io/WaterSortScreenshotSolver/)

This is a **static single-page app** that:

1. Parses a Water-Sort style puzzle screenshot in the browser using OpenCV.js (WASM) in a Web Worker
2. Reads the bottom powerup badges (retries / shuffles / add-bottles) using Tesseract.js (WASM OCR)
3. Runs a BFS solver to separate colors into single-color bottles

<img width="1885" height="930" alt="image" src="https://github.com/user-attachments/assets/89cf8e67-e4e2-49e2-90c8-9a1d5d62a6f9" />
