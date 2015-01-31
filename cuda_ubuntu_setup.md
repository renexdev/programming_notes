
sudo dpkg -i cuda-repo-ubuntu1404_6.5-14_amd64.deb 
sudo apt-get update

sudo apt-get install cuda


Set up environment variables. 
```bash
export PATH=/usr/local/cuda-6.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:$LD_LIBRARY_PATH
```

Check if it's working.

```shell
cuda-install-samples-6.5.sh  ~ 
cd ~/NVIDIA_CUDA-6.5_Samples 
make
```

Now run deviceQuery.
