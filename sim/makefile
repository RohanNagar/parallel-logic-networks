
test: gateMatrix.cpp gateMatrix.h test.cpp
	g++ gateMatrix.cpp gateMatrix.h test.cpp -o test

testClean: test
	rm test

cuda: simulator.cu gateMatrix.cpp
	nvcc -arch=compute_35 -code=sm_35 simulator.cu -x cu gateMatrix.cpp -o simulator

cleanCuda: simulator
	rm simulator
