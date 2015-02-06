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

