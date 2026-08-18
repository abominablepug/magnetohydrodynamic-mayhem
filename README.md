# Magnetohydrodynamic Mayhem
The last in a series of simulations, magnetohydrodynamic-mayhem (MHDM for short), is a **wgpu-based simulation** written fully in **Rust and WGSL**. Unlike prior simulations, MHDM is fully self-contained and doesn't use any external software for rendering or caclulations. This makes MHDM both the most technically challenging simulation I've written to date as well as the most in-depth physics-wise. In simple terms, MHDM sets out to **simulate the complex world of magnetohydrodynamics** (MHD) in a way that is both understandable and thorough.

## What is Magnetohydrodynamics?
Magnetohydrodyanmics (MHD) is a field of physics concerned with the **behavior of magnetic fluids**, as the name suggests. Swirling vortexes of magnetic fields interfere with fluid motion creating the chaotic and sharp movements of MHD substances. It's use can be most often found when predicting and simulating plasma, liquid metals, and, under certain conditions, salt water.

## The Simulated Mayhem
Simulating the intricacies of, not only an **incompressible fluid**, but **a magnetic one** can't be understated. In fact, when creating any computer simulation, it's important to find that perfect balance between accuracy, performance, and clarity. Accuracy comes first and foremost as, "premature optimization is the root of all evil" (Donald Knuth), so I began with writing the compute pipelines that'd become the backbone of the entire simulation. The simulation is built on a **2D staggered MAC (Marker-and-Cell) grid** which stores the magnetic field vector, velocity vector, and pressure within each cell. **MAC grids** are exceptional at tracking the movements of fluids making them the clear choice while the **staggered design** while using **bilinear interpolation** allows for fine-combed precision that prevents computational drift. By using a **Semi-Lagrangian scheme** to update cells, cell velocities and magnetic fields are updated each step. However, this also introduces computational drift as the simulation has to ensure that both the **fluid velocities** and **magnetic fields** remain **divergence-free** since the fluid is incompressible and monopoles can't exist. This is done using a **repeated Jacobi step** with a **Ping-Pong scheme** which flips the input and output buffer repeatedly to improve performance while approaching the divergence-free values. These steps are then repeated for each frame building the mathematical basis for the simulation.

## Beauty in Chaos
It's important to not lose the forest for the trees. At the end of the day, simulations should be educational and, maybe even, fun which is why the best ones include visuals and controls. MHDM is no different, allowing the user to visualize the **fluid velocity** (blue), **magnetic field** (purple), **magnetic moments** (gold), **particles** (cyan), and **fluid movement/dye** (green). While it can be overwhelming at first I found that, after tinkering, it was effectively able to explain the movement of particles and dyes throughout the simulation. The LMB and RMB can also be used to **introduce noise** into the simulation which let the user test out more interactions and become a **captain of the magnetohydrodynamic seas**. All this to say that MHDM provides the user with visuals and controls to get involved and learn about the world of MHD.

## Formulas
At its core, magnetohydrodynamics combines the field of **hydrodynamics** with **electromagnetism**. By combining the **Navier-Stokes equations** from fluid dynamics with **Maxwell's equations of electromagnetism**, MHD has some interesting formulas that were required for this project:
<br>

$$
\frac{∂u}{∂t} + (u \cdot ∇)u = -\frac1p∇p + \frac1p(J \times B)
$$

$$
\frac{∂B}{∂t} = ∇ \times (u \times B) = (B \cdot ∇)u - (u \cdot ∇)B
$$

$$
J = ∇ \times B
$$

$$
∇ \cdot u = 0
$$

$$
∇ \cdot B = 0
$$

The first equation is the **Navier-Stokes momentum equation** with an added **lorentz force term** to account for the magnetic field. Overall, it includes the **advection term** $(u \cdot ∇)u$, the **pressure gradient** $-\frac1p∇p$, and the **lorentz force term** $\frac1p(J \times B)$. Each part of this equation had to be solved in separate pipelines to ensure the accurate movement of the fluid. The second equation is **Induction Equation** including both the **magnetic stretching term** $(B \cdot ∇)u$ and the **magnetic advection term** $(u \cdot ∇)B$. This equation is essential to updating the magnetic field each step with both terms being solved in sequence. The third equation solves for the **charge density** at a point in the grid and was used to solve for the **lorentz force**. The last two equations mathematically symbolize that both the fluid velocity and magnetic field can not diverge.

## Results
