####**Configuring the Kernel Launch**
```c
square<<<dim3(bx,by,bz), dim3(tx,ty,tz)>>>(d_out, d_in);
```
`dim3(bx,by,bz)` is the number of **blocks**, and `dim3(tx,ty,tz)` is the number of **threads** in a block.  
`tx*ty*tz` cannot exceed 1024 (or 512 on older GPUs.)  

If one of the parameters of `dim3` is not filled out, the default value is 1. Therefore, the following three are equivalent:  
```c
square<<<1, 64>>>(d_out, d_in);
square<<<dim3(1), dim3(64)>>>(d_out, d_in);
square<<<dim3(1,1,1), dim3(64,1,1)>>>(d_out, d_in);
```
Some of the parameters that can be accessed in a kernel:  
`threadIdx`: the ID of the current thread in a particular block.  
`blockDim`: the size of the current block.  
`blockIdx`: the ID of the current block.  
`gridDim`: the size of the entire grid.  
All of them have `.x`, `.y`, and `.z` members.  
For example, in `square<<<dim3(8,4,2), dim3(16,16)>>>(d_out, d_in)`, `gridDim.y` = 4, and `blockDim.z` = 0.

####**Communication Patterns**
These are different types of communication patterns:  
**Map**: The input index and the output index are the same.  
**Gather**: The input index has to be calculated by the thread.  
**Scatter**: The output index has to be calculated by the thread.  
**Stencil**: Tasks read input from a fixed neighborhood in an array. *Should generate a result for every element in the array.*  
**Transpose**:  Tasks re-order data elements in memory.  
See the following codes for some examples:  
```cpp
float out[], in[];
int i = threadIdx.x;
int j = threadIdx.y;
const float pi = 3.1415;
out[i] = pi * in[i];                                // This is map
out[i + j*128] = in[j + i*128];                     // This is transpose
if(i % 2){
  out[i-1] += pi * in[i]; out[i+1] += pi * in[i];   // This is scatter
  out[i] = (in[i-1] + in[i] + in[i+1]) * pi / 3.0f; // This is gather 
}                                                   // Neither of the above two is stencil, since
                                                    // stencil requires every element to have a result.
```
####**GPU Hardware**  
A GPU is composed of a number of **streaming multi-processors** (**SM**).  
An SM consists of many simple processors and memory.  
  
A **programmer** is responsible for **defining** thread blocks in software.  
A **GPU** is responsible for **allocating** thread blocks to hardware SMs.   
  
All the threads in a thread block may cooperate to solve a sub-problem.  
All the threads in the same SM **cannot** cooperate to solve a sub-problem. (Because they are on different blocks!)

A programmer **cannot** specify whether block X runs before, after, or alongside block Y.  
A programmer **cannot** specify which SM block X will be allocated to.  
In other words, CUDA makes few guarantees about **when & where** a block will run. This is an advantage of CUDA that ensures efficiency & scalability.  
  
For example, if we launch the following kernel with 16 blocks and 1 thread per block, there're totally **16!**(~21 trillion) kinds of output. They are totally random.  
```cpp
#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void hello(){
    printf("Hello world! I'm a thread in block %d\n", blockIdx.x);
}

int main(int argc,char **argv){
    // launch the kernel
    hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

    // force the printf()s to flush
    cudaDeviceSynchronize();

    return 0;
}
```
What CUDA **does** guarantee are:  
All threads in a block run on the same SM at the same time.  
All blocks in a kernel finish before any blocks from the next kernel runs.  

