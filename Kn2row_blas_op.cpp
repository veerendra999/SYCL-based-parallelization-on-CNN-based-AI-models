//%%writefile Kn2row_blas_op.cpp
#include<iostream>
#include<cstring>
#include<chrono>
#include <sycl/sycl.hpp>          
#include "oneapi/mkl/blas.hpp"  


using namespace sycl;
namespace mkl = oneapi::mkl;  
extern "C" {
    double kn2row(const unsigned char* image_data, int rows, int cols, const int channels, const int channels_out,
        const unsigned char* kernel_data,  int k_row, int k_col, int stride, int padding_x, int padding_y, int opDim_x, int opDim_y, 
        const unsigned char* output_data){
        
        float* image = new float[channels * rows * cols];
        std::memcpy(image, image_data, (channels * rows * cols) * sizeof(float));

        float* kernel = new float[channels * channels_out * k_row * k_col];
        std::memcpy(kernel, kernel_data, (channels * channels_out * k_row * k_col) * sizeof(float));

        auto start = std::chrono::high_resolution_clock::now();
        
        //Creating the input image matrix;
        float** kn2row_input_array = new float*[channels];
        int input_H = channels;
        int input_W = ((rows + 2 * padding_x) * (cols + 2 * padding_y));
        for(int c = 0; c<channels; c++){
            
            int inpItr = 0;
            kn2row_input_array[c] = new float[input_W];
            
            for(int i=0; i<padding_x*(cols + 2 * padding_y); i++){
                kn2row_input_array[c][i] = 0;
            }

            for(int i = padding_x; i<rows+padding_x; i++){
                for(int p=0; p<padding_y; p++){
                    kn2row_input_array[c][i*(cols + 2 * padding_y) + p] = 0;
                }
                for(int j=padding_y; j<cols+padding_y; j++){
                    kn2row_input_array[c][i * (cols + 2 * padding_y)+j] = image[c * (rows * cols) + inpItr];
                    inpItr++;
                }
                for(int p=cols+padding_y; p<(cols + 2 * padding_y); p++){
                    kn2row_input_array[c][i*(cols + 2 * padding_y) + p] = 0;
                }
            }

            for(int i=((cols + 2 * padding_y) * (rows + padding_x)); i<((cols + 2 * padding_y) * (rows + 2 * padding_x)); i++){
                kn2row_input_array[c][i] = 0;
            }

        }

        
        //Creating the kernel matrix
        float* kn2row_kernel_array[k_row * k_col * channels_out];
        for(int c=0; c<channels_out; c++){
            for(int i=0; i<k_row*k_col; i++){
                kn2row_kernel_array[(c * k_row * k_col) + i] = new float[channels];
                for(int j=0; j<channels; j++){
                    kn2row_kernel_array[(c * k_row * k_col) + i][j] = kernel[(channels * c * k_row * k_col) + (j * k_row * k_col) + i];
                }
            }
        }
        
        
        //Multiplying the kernel and input matrices
        queue q(property::queue::enable_profiling{});
        
        float *matrixA = sycl::malloc_shared<float>(k_row * k_col * channels_out * channels, q);
        float *matrixB = sycl::malloc_shared<float>(channels * input_W, q);
        float *matrixC = sycl::malloc_shared<float>(k_row * k_col * channels_out * input_W, q);
        
        for(int i=0; i<k_row * k_col * channels_out; i++){
            for(int j=0; j<channels; j++){
                matrixA[i * channels + j] = kn2row_kernel_array[i][j];
            }
        }
        
        for(int i=0; i<channels; i++){
            for(int j=0; j<input_W; j++){
                matrixB[i * input_W + j] = kn2row_input_array[i][j];
            }
        }
        
        
        // removing matrices that are not needed
        delete[] image;
        delete[] kernel;
        for (int i = 0; i < input_H; ++i) {
            delete[] kn2row_input_array[i];
        }
        delete[] kn2row_input_array;
        
        for(int i = 0; i < (k_row * k_col * channels_out); ++i){
            delete[] kn2row_kernel_array[i];
        }
        
        device device = q.get_device();
        
        sycl::event gemm_done;
        std::vector<sycl::event> gemm_dependencies;
        
        float alpha = 1.0, beta = 1.0;
    
        mkl::transpose transA = mkl::transpose::nontrans;
        mkl::transpose transB = mkl::transpose::nontrans;
        int ldA = channels, ldB = input_W, ldC = input_W;

        //DPC++ kernel for matrix multiplication
        gemm_done = mkl::blas::row_major::gemm(q, transA, transB, k_row * k_col * channels_out, input_W, channels, alpha, matrixA, ldA, matrixB, ldB, beta, matrixC, ldC, gemm_dependencies);
        
        float *productMatrix[channels_out * k_row * k_col];
        for (int i=0; i <(k_row * k_col * channels_out); i++ ){
            productMatrix[i] = new float[input_W];
            for(int j=0; j<input_W; j++ ){
                productMatrix[i][j] = matrixC[i * (input_W) + j];
            }
        }
        
        float shiftAdd[channels_out][opDim_x][opDim_y];
        
        auto stop = std::chrono::high_resolution_clock::now();
        

        //Calculating the time taken
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double executionTimeMicroseconds = duration.count();

        float reshape[channels_out][opDim_x * opDim_y];
        for(int c=0; c<channels_out; c++){
            for(int i=0; i<opDim_x; i++){
                for(int j=0; j<opDim_y; j++){
                    reshape[c][i*opDim_y + j] = shiftAdd[c][i][j];
                }
            }
        }
        
        for (int i = 0; i < channels_out; ++i) {
            memcpy((void*)(output_data + i * (opDim_x * opDim_y) * sizeof(float)), (void*)reshape[i], (opDim_x * opDim_y) * sizeof(float));
        }
        
        return executionTimeMicroseconds;
        
    }
}