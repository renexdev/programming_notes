####**Congiguring the Kernel Launch**
```c
square<<<M, N>>>(d_out, d_in);
```
M is the number of **blocks**, and N is the number of **threads** in a block.  
N cannot exceed 1024 (or 512 on older GPUs.)
