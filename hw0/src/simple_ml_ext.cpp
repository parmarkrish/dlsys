#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t batch_size = batch; // make naming more explicit
    for (uint32_t b = 0; b < m; b += batch_size) {
      float *Z_I = new float[batch_size * k]();
      for (uint32_t i = b; i < b + batch_size; i++) {
        for (uint32_t j = 0; j < k; j++) {
          for (uint32_t p = 0; p < n; p++) {
            Z_I[(i-b)*k + j] += X[i*n + p] * theta[p*k + j]; // Z_i (batch_size, k)
          }
        }
      }

      // np.exp(X_b[i].dot(theta))
      for (uint32_t i = 0; i < batch_size * k; i++) {
        Z_I[i] = exp(Z_I[i]);
      }

      // Z_I /= np.sum(Z_I, axis=1)[:, np.newaxis]
      for (uint32_t i = 0; i < batch_size; i++) {
        float norm_factor = 0;
        for (uint32_t j = 0; j < k; j++) {
          norm_factor += Z_I[i*k + j];
        }
        for (uint32_t j = 0; j < k; j++) {
          Z_I[i*k + j] /= norm_factor;
        }
      }

      // Z_I[np.arange(batch_size), y_b[i]] -= 1
      for (uint32_t i = 0; i < batch_size; i++) {
        Z_I[i*k + (uint32_t)y[b + i]] -= 1;
      }

      // theta_grad = X_b[i].T.dot(Z_I) (b, k)
      float *theta_grad = new float[n * k]();
      for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < k; j++) {
          for (uint32_t p = b; p < b + batch_size; p++) {
            theta_grad[i*k + j] += X[p*n + i] * Z_I[(p-b)*k + j]; // Z_i (batch_size, k)
          }
        }
      }

      // theta -= lr/batch_size * theta_grad
      for (uint32_t i = 0; i < n * k; i++) {
        theta[i] -= lr/batch_size * theta_grad[i];
      }

      delete Z_I[];
      delete theta_grad[];
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
