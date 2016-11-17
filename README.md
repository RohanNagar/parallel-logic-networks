# Gate-Level Simulation on a GPU

This research was done as our term Project for EE 382C: Multicore Computing, taught by Dr. Vijay Garg.

- [Abstract](#abstract)

## Abstract
The purpose of this project is to simulate a gate-level netlist on a GPU in order to introduce parallelism during simulation. Gate-level logic networks are currently simulated with software such as ModelSim and is usually based on event driven evaluations on a CPU. Our approach allows for multiple gate layers to be evaluated simultaneously to improve the overall speed of simulation. This project involves parsing synthesized EDF files, constructing a graph of the represented network, and then performing the simulation on a GPU. Our results show that this approach is more performant than software simulators provided by the Digital System Design course of the University of Texas at Austin, and our implementation can be used by students to speed up their development and testing processes.

