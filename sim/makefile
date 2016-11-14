
manual: gateMatrix.cpp manualMatrix.cpp
	g++ gateMatrix.cpp manualMatrix.cpp -o manualMatrix

manualClean: manualMatrix
	rm manualMatrix

cuda: simulator.cu gateMatrix.cpp Tokenizer.cpp
	nvcc -arch=compute_35 -code=sm_35 -include $(Design.h) simulator.cu -x cu gateMatrix.cpp Tokenizer.cpp -o simulator

cleanCuda: simulator
	rm simulator

simUser: simulator.cu gateMatrix.cpp Tokenizer.cpp
	nvcc -arch=compute_35 -code=sm_35 -include $(Design.h) -D USERINPUT simulator.cu -x cu gateMatrix.cpp Tokenizer.cpp -o simUser

cleanSimUser: simUser
	rm simUser  

