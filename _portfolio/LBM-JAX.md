---
title: "Simulating Fluids with Lattice Boltzmann Method with GPU support using JAX "
excerpt: "A Python fluid dynamics solver based on the Lattice Boltzmann Method using JAX as its computational backend.<br/><video src='LBM-JAX/rayleigh-benard.mp4' width='60%' controls></video>"
collection: portfolio
---

Project source code can be found on [Github](https://github.com/hlasco/rllbm)




<figure>
  <h3> Rayleigh-Taylor instability triggered by oscillating heat source </h3>
  <video src='rayleigh-benard.mp4' width='80%' controls></video>
  <figcaption style="text-align: left;">This visualization shows the Rayleigh-Taylor instability in action, triggered by a local heating of the left wall of a box. The heated fluid is less dense and rises, while the colder, denser fluid sinks, creating a mixing layer that grows over time due to the gravitational force. The heat source moves periodically, and its amplitude oscillates between positive and negative values, causing the mixing layer to develop different shapes and structures. The simulation demonstrates the complex and dynamic behavior of the Rayleigh-Taylor instability
  </figcaption>
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

The LBM is a numerical method for simulating fluid dynamics. In LBM, the fluid is represented as a set of "particles" moving through a lattice. The positions and velocities of the particles are updated at each point in the lattice according to a set of rules based on the Boltzmann equation.

The Boltzmann equation describes the statistical behavior of particles in a gas or fluid. It can be written as:

$$\frac{\partial f}{\partial t} + \mathbf{u} \cdot \nabla f = \Omega$$


where $f(\mathbf{u}, \mathbf{x}, t)$ is the distribution function, which tells us how many particles of fluid are moving with a certain velocity $\mathbf{u}$ at a certain point in space $\mathbf{x}$ and time $t$. $\nabla$ is the gradient operator, and $\Omega$ represents the collision term, which describes the interactions between the particles.

To simulate the fluid dynamics using LBM, we first discretize the Boltzmann equation in both space and time. We then define a lattice, which is a regular grid of points in space. At each lattice point, we define a set of discrete velocities $\mathbf{e}_i$, where $i$ ranges over a set of discrete values. These discrete velocities correspond to the possible directions that particles can move in the lattice.


<figure style="display: inline-block; width: 80%">
  <img src="lattice.png" width="40%" height="auto">
  <figcaption>The discrete velocities in the D2Q9 and D2Q5 lattices.</figcaption>
</figure>


Next, we define a set of particle distribution functions $f_i(\mathbf{x},t)$, where $\mathbf{x}$ is the position in the lattice and $t$ is time. Each distribution function $f_i(\mathbf{x},t)$ represents the number of particles moving in the direction $\mathbf{e}_i$ at the lattice point $\mathbf{x}$ and time $t$.

The evolution of the distribution functions is governed by the following equation:

$$f_i(\mathbf{x}+\mathbf{e}_i\Delta t,t+\Delta t) = f_i(\mathbf{x},t) - \frac{1}{\tau}\left[f_i(\mathbf{x},t) - f_i^{eq}(\mathbf{x},t)\right]$$

where $\Delta t$ is the time step, $\tau$ is a relaxation time that determines the rate at which the distribution functions approach their equilibrium values, and $f_i^{eq}(\mathbf{x},t)$ is the local equilibrium distribution function, which is given by:

$$f_i^{eq}(\mathbf{x},t) = w_i \rho(\mathbf{x},t)\left[1 + \frac{\mathbf{e}_i \cdot \mathbf{u}(\mathbf{x},t)}{c_s^2} + \frac{(\mathbf{e}_i \cdot \mathbf{u}(\mathbf{x},t))^2}{2c_s^4} - \frac{\mathbf{u}(\mathbf{x},t) \cdot \mathbf{u}(\mathbf{x},t)}{2c_s^2}\right]$$

where $w_i$ are weights that depend on the discrete velocity $\mathbf{e}_i$, $\rho(\mathbf{x},t)$ is the local density, $\mathbf{u}(\mathbf{x},t)$ is the local velocity, and $c_s$ is the speed of sound, which is a parameter that depends on the specific fluid being simulated.