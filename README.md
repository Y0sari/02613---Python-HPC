# 🔥 02613 Mini-Project: Wall Heating (High Performance Computing)

## 📌 Project Overview

This project explores a novel **Wall Heating** approach where inside walls of buildings are heated instead of using traditional radiators or floor heating. The heating effect is simulated using numerical methods on thousands of 2D building floorplans derived from the **Modified Swiss Dwellings** dataset.

The project combines:

- Computational simulation (Jacobi method)
- Performance profiling and optimization
- Parallelization (CPU & GPU)
- Scientific data visualization
- Final statistical analysis

---

## 🏗️ Objective

Simulate steady-state heat diffusion in buildings with:

- **Inside walls** at **25°C**
- **Load-bearing walls** at **5°C**
- **Room interiors** initialized at 0°C

Then evaluate the temperature distribution using key metrics:

- 🔹 Mean temperature inside rooms
- 🔹 Temperature standard deviation
- 🔹 % area above 18°C (mold risk zone)
- 🔹 % area below 15°C (comfort threshold)

---

## 📁 Dataset Description

📂 Path: `/dtu/projects/02613_2025/data/modified_swiss_dwellings/`

Each building has:

- `{id}_domain.npy` → Grid with walls (5 or 25°C) and interior (0°C)
- `{id}_interior.npy` → Binary mask (1 = interior, 0 = wall/outside)

Additional file:

- `building_ids.txt` → List of all building IDs (≈4571 buildings)

Grid size: `512 x 512` (with padding to `514 x 514` during simulation)

---

## 🧮 Simulation Method: Jacobi Iteration

Solves the discrete Laplace equation to model steady-state heat flow:

```python
u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
```
