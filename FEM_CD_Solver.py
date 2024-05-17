################################################################
# Python code to solve the Convection Diffusion Equation in 1D #
# Author: Abhinav Jha                                          #
# Email : jha.abhinav0207@gmail.com                            #
################################################################

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

def assemble_local(f, diff_coeff, conv_coeff, xi, xi_1, degree):
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
    LM_Diff = np.zeros((degree + 1, degree + 1))
    LM_Conv = np.zeros((degree + 1, degree + 1))
    for i in range(degree + 1):
        for j in range(degree + 1):
            # [Diffusion]_ij = epsilon*(der_phi_i, der_phi_j)
            LM_Diff[i, j] = sp.integrate.quad(lambda x: diff_coeff*der_phi[i](x)*der_phi[j](x), xi, xi_1)[0]
            # [Convection]_ij = (b*der_phi_j, phi_i)
            LM_Conv[i, j] = sp.integrate.quad(lambda x: conv_coeff*der_phi[j](x)*phi[i](x), xi, xi_1)[0]
    Lb = np.zeros(degree + 1)
    Lb = np.array([sp.integrate.quad(lambda x: phi[i](x) * f(x), xi, xi_1)[0] for i in range(degree + 1)])

    return LM_Diff, LM_Conv, Lb

def assemble(grid, f, diff_coeff, conv_coeff, degree):
    num_nodes = len(grid) - 1  # number of elements
    K_Diff = lil_matrix((degree*num_nodes + 1, degree*num_nodes + 1))
    K_Conv = lil_matrix((degree*num_nodes + 1, degree*num_nodes + 1))
    b = np.zeros(degree*num_nodes + 1)
    for j in range(num_nodes):
        local_matrix_Diff, local_matrix_Conv, local_load = assemble_local(f,
                                                            diff_coeff, conv_coeff, grid[j], grid[j+1], degree)
        i = degree*j
        b[i:i + degree + 1] += local_load
        for m in range(degree + 1):
            for n in range(degree + 1):
                K_Diff[i + m, i + n] += local_matrix_Diff[m][n]
                K_Conv[i + m, i + n] += local_matrix_Conv[m][n]
    return K_Diff, K_Conv, b

def solve(grid, f, diff_coeff, conv_coeff, u_b, degree):
    K_Diff, K_Conv, b = assemble(grid, f, diff_coeff, conv_coeff, degree)

    # Add Diffusion and Convection matrix
    K = K_Diff + K_Conv
    num_nodes = len(grid) - 1

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

def analytical_solution(x, diff_coeff):
    return (np.exp((x-1)/diff_coeff) - np.exp(-1/diff_coeff))/(1-np.exp(-1/diff_coeff))

def compute_error(nodes, diff_coeff, u_num):
    def integrand(x):
        u_analytical = analytical_solution(x, diff_coeff)
        u_num_interp = np.interp(x, nodes, u_num)
        return (u_num_interp - u_analytical)**2

    def numerical_gradient(x):
        return np.gradient(u_num, nodes)

    def analytical_gradient(x):
        return np.exp((x-1)/diff_coeff) / (diff_coeff * (1-np.exp(-1/diff_coeff)))

    def gradient_error_integrand(x):
        num_grad = np.interp(x, nodes, numerical_gradient(x))
        ana_grad = analytical_gradient(x)
        return (num_grad - ana_grad)**2

    l2_error = sp.integrate.quad(integrand, 0, 1)[0]
    l2_grad_error = sp.integrate.quad(gradient_error_integrand, 0, 1)[0]

    return np.sqrt(l2_error), np.sqrt(l2_grad_error)


def main():
    # Define parameters
    a, b = 0, 1  # Domain [a, b]
    n = 1000  # Number of elements
    u_b = [0, 1] # Dirchlet Boundary Condition
    degree = 1
    diff_coeff = 1e-1
    conv_coeff = 1

    # Generate mesh
    grid = np.linspace(a, b, n+1)
    x_plot = np.linspace(a, b, degree*n+1)

    # Define the function f(x)
    def f(x):
        return 0

    # Solve the problem
    u_numerical = solve(grid, f, diff_coeff, conv_coeff, u_b, degree)

    # Compute the true solution
    x_analytical = np.linspace(0, 1, 100)
    u_analytical = analytical_solution(x_analytical, diff_coeff)

    # Compute the L^2 error
    l2_error, l2_grad_error = compute_error(x_plot, diff_coeff, u_numerical)
    print("L^2 :", l2_error)
    print("H^1 :", l2_grad_error)

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
