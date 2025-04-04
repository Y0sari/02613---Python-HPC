# 02613 Mini-Project: Wall Heating!

## 🔥 Overview

This project evaluates a fictional heating system called **Wall Heating**, where inside walls of buildings are heated to 25°C, while load-bearing walls stay cold at 5°C. The system is simulated across thousands of 2D building floorplans using the **Jacobi method** to compute steady-state temperature distributions.

The dataset comes from the **Modified Swiss Dwellings** project and contains over 4500 labeled floorplans. The goal is to:

- Simulate temperature distributions in buildings
- Measure the effectiveness of the Wall Heating strategy
- **Optimize** and **accelerate** the simulation through various methods

---

## 🧪 Simulation Approach

We solve the steady-state heat distribution governed by **Laplace’s equation** with Dirichlet boundary conditions:

- Load-bearing walls: fixed at **5°C**
- Inside walls: fixed at **25°C**
- Interior points (rooms): iteratively updated using the **Jacobi method**

### Jacobi Iteration Formula:

For each interior point `u[i, j]`:
