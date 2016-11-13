
test: gateMatrix.cpp gateMatrix.h manualMatrix.cpp
	g++ gateMatrix.cpp gateMatrix.h manualMatrix.cpp -o manualMatrix

testClean: manualMatrix
	rm manualMatrix

cuda: simulator.cu gateMatrix.cpp
	nvcc -arch=compute_35 -code=sm_35 simulator.cu -x cu gateMatrix.cpp -o simulator

cleanCuda: simulator
	rm simulator
