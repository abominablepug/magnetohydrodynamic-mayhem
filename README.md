# Magnetohydrodynamic Mayhem
The last in a series of simulations, magnetohydrodynamic-mayhem (MHDM for short), is a **wgpu-based simulation** written fully in **Rust and WGSL**. Unlike prior simulations, MHDM is fully self-contained and doesn't use any external software for rendering or calculations. This makes MHDM both the most technically challenging simulation I've written to date as well as the most in-depth physics-wise. In simple terms, MHDM sets out to **simulate the complex world of magnetohydrodynamics** (MHD) in a way that is both understandable and thorough.

<img width="960" height="540" alt="MHDM" src="https://github.com/user-attachments/assets/287bcbc3-243e-4ca2-9620-5c6687694180" />

## What is Magnetohydrodynamics?
Magnetohydrodynamics (MHD) is a field of physics concerned with the **behavior of magnetic fluids**, as the name suggests. Swirling vortexes of magnetic fields interfere with fluid motion creating the chaotic and sharp movements of MHD substances. It's use can most often be found when predicting and simulating plasma, liquid metals, and, under certain conditions, salt water.

## The Simulated Mayhem
Simulating the intricacies of, not only an **incompressible fluid**, but **a magnetic one** cannot be overstated. In fact, when creating any computer simulation, it's important to find that perfect balance between accuracy, performance, and clarity. Accuracy comes first and foremost as, "premature optimization is the root of all evil" (Donald Knuth), so I began with writing the compute pipelines that'd become the backbone of the entire simulation. The simulation is built on a **2D staggered MAC (Marker-and-Cell) grid** which stores the magnetic field vector, velocity vector, and pressure within each cell. **MAC grids** are exceptional at tracking the movements of fluids making them the clear choice and the **staggered design**, combined with **bilinear interpolation**, allows for fine-grained precision that prevents computational drift. By using a **Semi-Lagrangian scheme** to update cells, cell velocities and magnetic fields are updated each step. However, this also introduces computational drift as the simulation has to ensure that both the **fluid velocities** and **magnetic fields** remain **divergence-free** since the fluid is incompressible and monopoles can't exist. This is done using a **repeated Jacobi step** with a **Ping-Pong scheme** which flips the input and output buffer repeatedly to improve performance while approaching the divergence-free values. These steps are then repeated for each frame building the mathematical basis for the simulation.

<img width="2559" height="1599" alt="screenshot_2026-08-18_05-58-02" src="https://github.com/user-attachments/assets/6079749d-88f1-480f-bb57-14fbd7946b5c" />

## Beauty in Chaos
It's important to not lose the forest for the trees. At the end of the day, simulations should be educational and, maybe even, fun which is why the best ones include visuals and controls. MHDM is no different, allowing the user to visualize the **fluid velocity** (blue), **magnetic field** (purple), **magnetic moments** (gold), **particles** (cyan), and **fluid movement/dye** (green). While it can be overwhelming at first I found that, after tinkering, it was effectively able to explain the movement of particles and dyes throughout the simulation. The LMB and RMB can also be used to **introduce noise** into the simulation which lets the user test out more interactions and become a **captain of the magnetohydrodynamic seas**. All this to say that MHDM provides the user with visuals and controls to get involved and learn about the world of MHD.

<img width="2554" height="1599" alt="screenshot_2026-08-18_05-53-16" src="https://github.com/user-attachments/assets/a73ef131-5830-4679-93a9-387e52bcf050" />

## Formulas
At its core, magnetohydrodynamics combines the field of **hydrodynamics** with **electromagnetism**. By combining the **Navier-Stokes equations** from fluid dynamics with **Maxwell's equations of electromagnetism**, MHD has some interesting formulas that were required for this project:
<br>

$$
\frac{\partial \vec{u}}{\partial t} + (\vec{u} \cdot \nabla)\vec{u} = -\frac{1}{\rho}\nabla p + \frac{1}{\rho}(\vec{J} \times \vec{B})
$$

$$
\frac{\partial \vec{B}}{\partial t} = \nabla \times (\vec{u} \times \vec{B}) = (\vec{B} \cdot \nabla)\vec{u} - (\vec{u} \cdot \nabla)\vec{B}
$$

$$
\vec{J} = \nabla \times \vec{B}
$$

$$
\nabla \cdot \vec{u} = 0
$$

$$
\nabla \cdot \vec{B} = 0
$$

The first equation is the **Navier-Stokes momentum equation** with an added **Lorentz force term** to account for the magnetic field. Overall, it includes the **advection term** $(u \cdot ∇)u$, the **pressure gradient** $-\frac1p∇p$, and the **Lorentz force term** $\frac1p(J \times B)$. Each part of this equation had to be solved in separate pipelines to ensure the accurate movement of the fluid. The second equation is **Induction Equation** including both the **magnetic stretching term** $(B \cdot ∇)u$ and the **magnetic advection term** $(u \cdot ∇)B$. This equation is essential to updating the magnetic field each step with both terms being solved in sequence. The third equation solves for the **current density** at a point in the grid and was used to solve for the **Lorentz force**. The last two equations mathematically symbolize that both the fluid velocity and magnetic field can not diverge.

## How to use
By downloading your respective executable file you can run the simulation on your own computer and test it out for yourself!

<img width="2559" height="1599" alt="screenshot_2026-08-18_05-58-24" src="https://github.com/user-attachments/assets/030f37bc-b721-4d64-bf51-e7e8d6e4a9cd" />
