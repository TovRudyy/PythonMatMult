CC=c++
NAME=matmul
EXTRAE="-I${EXTRAE_HOME}/include -L${EXTRAE_HOME}/lib -lompitrace"
matmul: matmul.cpp
	$(CC) -g -finstrument-functions -O3 -fopenmp -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` $(NAME).cpp -o $(NAME)`python3-config --extension-suffix`

clean:
	rm -f *.so
