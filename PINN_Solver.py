###############################################################
# Python code to solve the Poisson Equation in 1D using PINNs #
# Author: Abhinav Jha                                         #
# Email : jha.abhinav0207@gmail.com                           #
###############################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create a class called PINN which inherits from tf.keras.Model
class PINN(tf.keras.Model):
    def __init__(self, n):
        # super is calling the constructor of the parent class keras
        super(PINN, self).__init__()
        # Hidden layer with n inputs, tanh activation function
        self.dense1 = tf.keras.layers.Dense(n, activation='tanh')
        # Output layer with 1 neuron
        self.dense2 = tf.keras.layers.Dense(1)

    # Forward passing, convert 'inputs' to a tensor, pass through the hidden
    # layer dense1 and then to the dense2 layer
    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        x = self.dense1(x)
        return self.dense2(x)

# Define the loss function
def loss(model, x):
    # Compute the gradient, nested loop is required for computing the second
    # derivative
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            # watch is used to track for x
            tape1.watch(x)
            tape2.watch(x)
            # Compute the predicted solution
            u = model(x)
        # Compute the first gradient
        u_x = tape1.gradient(u, x)
    # Compute the second gradient
    u_xx = tape2.gradient(u_x, x)
    # Differential equation residual
    f = u_xx + np.pi**2*np.sin(np.pi*x)
    # Boundary condition loss, u[-1] corresponds to u(1)
    bc_loss = tf.square(u[0] - 0) + tf.square(u[-1] - 0)
    # Equation residual loss, in the form of mean squared residual
    eq_loss = tf.reduce_mean(tf.square(f))
    return eq_loss + bc_loss

# Training of the model
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        # Compute the loss
        loss_value = loss(model, x)
    # Gradient of the loss
    gradients = tape.gradient(loss_value, model.trainable_variables)
    # Update the model variable
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value

# Generate training data
# 100 points between 0 and 1, [:, None] converts to a column vector
x_train = np.linspace(0, 1, 100)[:, None]

# Convert x_train to TensorFlow tensor
x_train_tf = tf.convert_to_tensor(x_train, dtype=tf.float32)

# Initialize PINN with 20 neurons
model = PINN(n=20)
# Adam optimizer with learning rate of 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
epochs = 5000
for epoch in range(epochs):
    loss_value = train_step(model, x_train_tf, optimizer)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value}")

# Generate test data for plotting
x_test = np.linspace(0, 1, 1000)[:, None]  # More points for smoother plot

# Predict u(x) using the trained model
u_pred = model(x_test)

# True solution
def analytical_solution(x):
    return np.sin(np.pi*x)

# Compute the true solution
x_analytical = np.linspace(0, 1, 10)
u_analytical = analytical_solution(x_analytical)

# Plot the solution
plt.figure(figsize=(8, 6))
plt.plot(x_test, u_pred, label='Predicted Solution')
plt.plot(x_analytical, u_analytical, label='True solution', color='red', marker = "o")
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('PINN Solution to u\'\'(x) = 6x')
plt.legend()
plt.grid(True)
plt.show()
