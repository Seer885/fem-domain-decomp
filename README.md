Currently working on applying Domain Decomposition (DD) to existing FEM scripts to meaningfully reduce their runtime. The main advantage of DD is that it splits the singular domain of a FEM script into multiple subdomains, which allows for the usage of parallel computing to significantly reduce the runtime.
DD has two distinct components: Iterative solving and preconditioning. The former has been successfully implemented and involves the following steps under the hood:
1. Split the domain into n subdomains
2. Use the CG iterative solver to find the unknown boundary values on each subdomain
3. Find the interior (non-boundary) values on each subdomain using a more standard FEM script with Dirichlet boundary conditions
Any step that involves work being done "on each subdomain" can be done with parallel computing; however, this option was unfortunately not available while this project was being worked on.

As for the latter component of DD, this is currently a work in a progress and has not yet been successfully implemented- Future commits will be performed once this changes.

To help make debugging as easy as possible, each component of DD was coded in stages with small tweaks between each one.
0) FEM was applied to the basic 2D Poisson equation with uniform Q1 elements and no iterative solving whatsoever. This exists for the sake of verifying that later (more complicated) stages work correctly by comparing the errors resulting from identical input parameters.
1) An extremely naive implementation of DD was applied to the previous FEM script with little work being divided between subdomains. This script exists for a similar purpose as the previous one.
2) Much of the work has now been split across the n subdomains, though CG has not yet been implemented.
3) Some minor optimizations were performed, trimming off around 20 lines and reducing the storage costs slightly.
4) CG over each subdomain was properly implemented, fully avoiding the need to explicitly compute the Schur compliment on each subdomain (a normally expensive operation).

Once both components of DD have been applied to the 2D Poisson PDE, the plan will be to apply them to more challenging PDE's, including the steady Navier Stokes equation.
