###################################################
# Python code to solve the Poisson Equation in 1D #
# Author: Abhinav Jha                             #
# Email : jha.abhinav0207@gmail.com               #
###################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import lagrange
from scipy import integrate

def lagrange_polynomials(x, degree):
    y = np.array([1] + [0] * degree)
    poly = lagrange(x, y)
    return poly

def assemble_local(f, xi, xi_1, degree):
    h = (xi_1 - xi)/degree
    x = np.zeros(degree + 1)
    phi = []
    der_phi = []
    for i in range(degree + 1):
        x[i] = xi + i*h
    for i in range(degree + 1):
        x_input = np.zeros(degree + 1)
        for j in range(degree + 1):
            x_input[j] = x[(j+i)%(degree+1)]
        phi.append(lagrange_polynomials(x_input, degree))
        der_phi.append(np.polyder(phi[i]))
    LM = np.zeros((degree + 1, degree + 1))
    for i in range(degree + 1):
        for j in range(degree + 1):
            LM[i, j] = sp.integrate.quad(lambda x: der_phi[i](x) * der_phi[j](x), xi, xi_1)[0]
    Lb = np.zeros(degree + 1)
    Lb = np.array([sp.integrate.quad(lambda x: phi[i](x) * f(x), xi, xi_1)[0] for i in range(degree + 1)])

    return LM, Lb

def assemble(xi, f, degree):
    num_nodes = len(xi) - 1  # number of elements
    K = lil_matrix((degree*num_nodes + 1, degree*num_nodes + 1))
    b = np.zeros(degree*num_nodes + 1)
    for j in range(num_nodes):
        local_matrix, local_load = assemble_local(f, xi[j], xi[j+1], degree)
        i = degree*j
        b[i:i + degree + 1] += local_load
        for m in range(degree + 1):
            for n in range(degree + 1):
                K[i + m, i + n] += local_matrix[m][n]
    return K, b

def solve(xi, f, u_b, degree):
    K, b = assemble(xi, f, degree)

    num_nodes = len(xi) - 1

    # Create the submatrix
    K_Active = lil_matrix((degree*num_nodes - 1, degree*num_nodes - 1))
    b_Active = np.zeros(degree*num_nodes - 1)
    K_Active = K[1:-1, 1:-1]
    b_Active = b[1:-1]
    # Adjust Dirchlet entries
    for i in range(degree):
        b_Active[i] -= K[i+1,0]*u_b[0]
        b_Active[-1-i] -= K[-2-i,-1]*u_b[1]

    u_active = np.zeros(degree*num_nodes - 1)
    # Solve the linear system
    u_active = spsolve(K_Active, b_Active)
    u = np.zeros(degree*num_nodes + 1)
    u[1:-1] = u_active

    # Set the Dirichlet DOFs
    u[0] = u_b[0]
    u[-1] = u_b[1]

    return u

def analytical_solution(x):
    return x**3

def compute_l2_error(nodes, u_num):
    def integrand(x):
        u_analytical = analytical_solution(x)
        u_num_interp = np.interp(x, nodes, u_num)
        return (u_num_interp - u_analytical)**2

    l2_error = sp.integrate.quad(integrand, 0, 1)[0]
    return np.sqrt(l2_error)

def main():
    # Define parameters
    a, b = 0, 1  # Domain [a, b]
    n = 50  # Number of elements
    u_b = [0, 1] # Dirchlet Boundary Condition
    degree = 4

    # Generate mesh
    xi = np.linspace(a, b, n+1)
    x_plot = np.linspace(a, b, degree*n+1)

    # Define the function f(x)
    def f(x):
        return -6*x

    # Solve the problem
    u_numerical = solve(xi, f, u_b, degree)

    # Compute the true solution
    x_analytical = np.linspace(0, 1, 100)
    u_analytical = analytical_solution(x_analytical)

    # Compute the L^2 error
    l2_error = compute_l2_error(x_plot, u_numerical)
    print("L^2 Error:", l2_error)

    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, u_numerical, label='Numerical Solution', color='blue', marker = "o")
    plt.plot(x_analytical, u_analytical, label='True solution', color='red')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Finite Element Method Solution for u\'\'(x) = sin(pi*x)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
