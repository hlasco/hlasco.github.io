---
title: "Simulating Fluids with Lattice Boltzmann Method with GPU support using JAX "
excerpt: "A Python fluid dynamics solver based on the Lattice Boltzmann Method using JAX as its computational backend.<br/><embed type='text/html' src='LBM-JAX/rayleigh-benard.html' height='210'>"
collection: portfolio
---

Project source code can be found on [Github](https://github.com/hlasco/rllbm)
<figure style="display: flex; flex-direction: row; justify-content: space-between; overflow: hidden;">
  <h3> Rayleigh-Taylor instability triggered by oscillating heat source </h3>
  <embed type="text/html" src="rayleigh-benard.html" height="210">
  <figcaption style="text-align: left;">This visualization shows the Rayleigh-Taylor instability in action, triggered by a local heating of the left wall of a box. The heated fluid is less dense and rises, while the colder, denser fluid sinks, creating a mixing layer that grows over time due to the gravitational force. The heat source moves periodically, and its amplitude oscillates between positive and negative values, causing the mixing layer to develop different shapes and structures. The simulation demonstrates the complex and dynamic behavior of the Rayleigh-Taylor instability</figcaption>
</figure>

Computational fluid dynamics (CFD) is an important field of study that has applications in many areas, including engineering, physics, and chemistry. One of the popular techniques to simulate fluids is the Lattice Boltzmann Method (LBM). This method involves simulating the motion of fluid particles using statistical mechanics principles.

One of the advantages of the LBM is that it's highly parallelizable and can be efficiently implemented on modern hardware like GPUs. Another advantage is that it can handle complex geometries and boundary conditions, making it well-suited for simulating fluid dynamics in a wide range of applications.

When it comes to implementing a LBM code, I've found that JAX is an excellent choice for the following reasons:
1. **Pythonic interface**: JAX provides a user-friendly Pythonic interface, making it easy to write and debug simulation code. This is especially useful for developpers who are not familiar with low-level programming languages like C++.
1. **Just-in-time (JIT) compilation**: JAX uses a JIT compilation process that can optimize the computations by compiling the code at runtime. This can lead to significant speedups.
1. **Efficient on modern hardware**: JAX provides built-in support for GPU acceleration, which can significantly speed up the computations involved in the LBM. This means that simulations can be run faster and more efficiently, which is especially important when dealing with large-scale simulations.
1. **Automatic differentiation**: JAX provides automatic differentiation, which can be used to compute gradients of the simulation with respect to different parameters. This is useful for optimization problems and can be used to tune simulation parameters for optimal results.

Lattice Boltzmann Method
======

The LBM simulates fluid dynamics by breaking up the fluid into a set of "particles" and tracking their positions and velocities through a lattice over time. The positions and velocities of the particles are updated at each point in the lattice according to a set of rules based on the Boltzmann equation, which describes the statistical behavior of particles in a gas or fluid. The fluid is modeled using a distribution function that tells us how many particles of fluid are moving in a certain direction at a certain point in space. By updating the distribution function, we can simulate how the fluid flows and behaves in different situations.
