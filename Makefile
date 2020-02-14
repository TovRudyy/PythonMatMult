CC=c++

matmul: matmul.cpp
	$(CC) -fopenmp -O3 -Wall -shared -std=c++11 -fPIC -I/usr/include/python3.7m -I/usr/local/include/python3.7m -I/home/orudyy/.local/include/python3.7m matmul.cpp -o matmul.cpython-37m-x86_64-linux-gnu.so

clean:
	rm -f *.so
