---
title: "Simulating Fluids with Lattice Boltzmann Method with GPU support using JAX "
excerpt: "A Python fluid dynamics solver based on the Lattice Boltzmann Method using JAX as its computational backend.<br/><embed type='text/html' src='LBM-JAX/rayleigh.html' width='550' height='300'>"
collection: portfolio
---

Computational fluid dynamics (CFD) is an important field of study that has applications in many areas, including engineering, physics, and chemistry. One of the popular techniques to simulate fluids is the Lattice Boltzmann Method (LBM). This method involves simulating the motion of fluid particles using statistical mechanics principles.

One of the advantages of the LBM is that it's highly parallelizable and can be efficiently implemented on modern hardware like GPUs. Another advantage is that it can handle complex geometries and boundary conditions, making it well-suited for simulating fluid dynamics in a wide range of applications.

When it comes to implementing a LBM code, I've found that JAX is an excellent choice for the following reasons:
1. **Pythonic interface**: JAX provides a user-friendly Pythonic interface, making it easy to write and debug simulation code. This is especially useful for developpers who are not familiar with low-level programming languages like C++.
1. **Just-in-time (JIT) compilation**: JAX uses a JIT compilation process that can optimize the computations by compiling the code at runtime. This can lead to significant speedups.
1. **Efficient on modern hardware**: JAX provides built-in support for GPU acceleration, which can significantly speed up the computations involved in the LBM. This means that simulations can be run faster and more efficiently, which is especially important when dealing with large-scale simulations.
1. **Automatic differentiation**: JAX provides automatic differentiation, which can be used to compute gradients of the simulation with respect to different parameters. This is useful for optimization problems and can be used to tune simulation parameters for optimal results.

Lattice Boltzmann Method
======

The LBM simulates fluid dynamics by breaking up the fluid into a set of "particles" and tracking their positions and velocities through a lattice over time. The positions and velocities of the particles are updated at each point in the lattice according to a set of rules based on the Boltzmann equation, which describes the statistical behavior of particles in a gas or fluid. The fluid is modeled using a distribution function that tells us how many particles of fluid are moving in a certain direction at a certain point in space. We use this function to simulate the movement of fluids by updating the distribution function at each point in the lattice over time. By updating the distribution function, we can simulate how the fluid flows and behaves in different situations. Essentially, the distribution function is a way to keep track of how much fluid is moving in which direction at each point in space.


The distribution function for a fluid particle with velocity $\mathbf{v}$ and position $\mathbf{x}$ at time $t$ is denoted by $f(\mathbf{x}, \mathbf{v}, t)$.

The Lattice Boltzmann Method equations are:

$$\begin{align*}
f_i(\mathbf{x} + \mathbf{v}_i\Delta t, t+\Delta t) - f_i(\mathbf{x}, t) = -\frac{\Delta t}{\tau}(f_i(\mathbf{x}, t) - f_i^{eq}(\mathbf{x}, t)) + \Delta t F_i(\mathbf{x}, t) \\
f_i^{eq}(\mathbf{x}, t) = w_i \rho(\mathbf{x}, t) \left[1 + \frac{\mathbf{v}_i \cdot \mathbf{u}(\mathbf{x}, t)}{c_s^2} + \frac{(\mathbf{v}_i \cdot \mathbf{u}(\mathbf{x}, t))^2}{2c_s^4} - \frac{|\mathbf{u}(\mathbf{x}, t)|^2}{2c_s^2}\right]
\end{align*}$$

where:
- $f_i(\mathbf{x}, t)$ is the distribution function for the $i$-th velocity direction at position $\mathbf{x}$ and time $t$.
- $\Delta t$ is the time step.
- $\tau$ is the relaxation time, which controls the viscosity of the fluid.
- $F_i(\mathbf{x}, t)$ is the force acting on the fluid particle with velocity $\mathbf{v}_i$ at position $\mathbf{x}$ and time $t$.
- $w_i$ is the weight associated with the $i$-th velocity direction.
- $\rho(\mathbf{x}, t)$ is the density of the fluid at position $\mathbf{x}$ and time $t$.
- $\mathbf{u}(\mathbf{x}, t)$ is the macroscopic velocity of the fluid at position $\mathbf{x}$ and time $t$.
- $c_s$ is the speed of sound, which is related to the lattice spacing and time step.
- $f_i^{eq}(\mathbf{x}, t)$ is the equilibrium distribution function for the $i$-th velocity direction at position $\mathbf{x}$ and time $t$.

The equilibrium distribution function is calculated using the density and velocity of the fluid at each lattice point, and depends on the specific lattice used in the simulation.

<figure style="display: flex; flex-direction: row; justify-content: space-between; overflow: hidden;">
  <h2>The Rayleigh Benard Instability</h2>
  <embed type="text/html" src="rayleigh.html" height="220">
  <figcaption style="text-align: right;">Caption for the visualization goes here.</figcaption>
</figure>