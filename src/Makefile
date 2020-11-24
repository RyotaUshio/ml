CXX = g++
CXXFLAGS = -std=c++17 -O2

.PHONY: all clean

all: main

main: main.o neuron.o
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm $(wildcard *.o)
