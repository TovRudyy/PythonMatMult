#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Python.h>
#include<iostream>
#include<omp.h>

namespace py = pybind11;

// py:array_t is to work with a Numpy array
py::array_t<int32_t> matmul(py::array_t<int32_t> A, py::array_t<int32_t> B, std::size_t chunk) {
    // I must work with a buffer_object containing all the information of the array
    py::buffer_info A_buf= A.request();
    py::buffer_info B_buf = B.request();
    std::size_t cols = A_buf.shape[1];
    // I create the result array
    py::array_t<int32_t> res = py::array_t<int32_t>(chunk*cols);
    py::buffer_info res_buf = res.request();
    // I get the point32_ters to the data
    int32_t *A_vals = (int32_t*) A_buf.ptr, *B_vals = (int32_t*) B_buf.ptr, *res_vals = (int32_t*) res_buf.ptr;

    for (std::size_t i = 0; i < chunk; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            res_vals[i*cols + j] = 0;
            for (std::size_t k = 0; k < cols; k++) {
                res_vals[i*cols + j] += A_vals[i*cols+ k] * B_vals[k*cols + j];
            }
        }
    }
    // I resize the result array to fit the correct shape
    res.resize({chunk, cols});
    return res;
}

// py:array_t is to work with a Numpy array
py::array_t<int32_t> matmul_omp(py::array_t<int32_t> A, py::array_t<int32_t> B, std::size_t chunk) {
    // I must work with a buffer_object containing all the information of the array
    py::buffer_info A_buf= A.request();
    py::buffer_info B_buf = B.request();
    std::size_t cols = A_buf.shape[1];
    // I create the result array
    py::array_t<int32_t> res = py::array_t<int32_t>(chunk*cols);
    py::buffer_info res_buf = res.request();
    // I get the point32_ters to the data
    int32_t *A_vals = (int32_t*) A_buf.ptr, *B_vals = (int32_t*) B_buf.ptr, *res_vals = (int32_t*) res_buf.ptr;

    #pragma omp parallel
    {
        #pragma omp single nowait
        std::cerr << "Number of OMP threads: " << omp_get_num_threads() << "\n";
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < chunk; i++) {
            for (std::size_t j = 0; j < cols; j++) {
                res_vals[i*cols + j] = 0;
                for (std::size_t k = 0; k < cols; k++) {
                    res_vals[i*cols + j] += A_vals[i*cols+ k] * B_vals[k*cols + j];
                }
            }
        }
    }
    // I resize the result array to fit the correct shape
    res.resize({chunk, cols});
    return res;
}

PYBIND11_MODULE(matmul, m) {
    m.doc() = "pybind11 matrix multiplication plugin";
    m.def("matmul", &matmul, "Multiply 2 square matrix NxN");
    m.def("matmul_omp", &matmul_omp, "Multiply 2 square matrix NxN with OpenMP parallelization");
}