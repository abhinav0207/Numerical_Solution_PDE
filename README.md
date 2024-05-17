# Numerical_Solution_PDE

Welcome to the **Numerical_Solution_PDE** repository. This repository contains Python scripts for solving differential equations using: Finite Element Method (FEM) and Physics-Informed Neural Networks (PINNs).

## Table of Contents

- [Implemented Programs](#implemented-programs)
  - [FEM_Poisson_Solver.py](#fem_poisson_solverpy)
  - [FEM_CD_Solver.py](#fem_cd_solverpy)
  - [PINN_Solver.py](#pinn_solverpy)
- [Usage](#usage)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Implemented Programs

### FEM_Poisson_Solver.py

This script solves the Dirichlet Poisson Problem in 1D using the Finite Element Method (FEM).

- **Problem:** Solves \( -\frac{d^2u}{dx^2} = f \) with Dirichlet boundary conditions.

### FEM_CD_Solver.py

This script solves the Dirichlet Convection-Diffusion Problem in 1D using the Finite Element Method (FEM).

- **Problem:** Solves \( -\frac{d^2u}{dx^2} + \frac{d}{dx}(vu) = f \) with Dirichlet boundary conditions.
- **Features:** Highlights how the FEM may fail when convection dominates diffusion.

### PINN_Solver.py

This script solves the Dirichlet Poisson Problem in 1D using Physics-Informed Neural Networks (PINNs).

- **Problem:** Solves \( -\frac{d^2u}{dx^2} = f \) with Dirichlet boundary conditions using a neural network.
- **Features:** Demonstrates the use of PINNs for solving PDEs.

## Usage

To use these scripts, clone the repository and run the desired Python file. Ensure you have the necessary dependencies installed, which typically include NumPy, SciPy, and TensorFlow for the PINN implementation.

```sh
git clone https://github.com/yourusername/Numerical_Solution_PDE.git
cd Numerical_Solution_PDE
python3 FEM_Poisson_Solver.py
# or
python3 FEM_CD_Solver.py
# or
python3 PINN_Solver.py

## Citation

If you use these scripts in your research or project, please cite this repository. This helps others find and utilize this work.

```bibtex
@misc{abhinav0207/Numerical_Solution_PDE,
  author = {Abhinav Jha},
  title = {Numerical Solution of PDE},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/abhinav0207/Numerical_Solution_PDE}},
}
