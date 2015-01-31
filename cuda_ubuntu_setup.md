Retrieve the repository and install cuda.
```shell
sudo dpkg -i cuda-repo-ubuntu1404_6.5-14_amd64.deb 
sudo apt-get update
sudo apt-get install cuda
```
Set up environment variables. 
```bash
export PATH=/usr/local/cuda-6.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:$LD_LIBRARY_PATH
```
Copy the sample codes and run deviceQuery to see if it's working.
```shell
cuda-install-samples-6.5.sh  ~ 
cd ~/NVIDIA_CUDA-6.5_Samples 
make
```
The following is an example code from Udacity:
```c
#cubeNumber.cu
#include <stdio.h>

__global__ void cube(float * d_out, float * d_in){
	// Todo: Fill in this function
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f*f*f;
}

int main(int argc, char ** argv) {
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = float(i);
	}
	float h_out[ARRAY_SIZE];

	// declare GPU memory pointers
	float * d_in;
	float * d_out;

	// allocate GPU memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	cube<<<1, ARRAY_SIZE>>>(d_out, d_in);

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for (int i =0; i < ARRAY_SIZE; i++) {
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
```
To compile it, use the following command.
```shell
nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib cubeNumber.cu -o cubeNumber
```

References: 
[NVIDIA CUDA Getting Started Guide for Linux](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#introduction)
