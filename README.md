# NVIDIA Profiler  
Record CPU and GPU Performance at regular intervals using [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface), a command line utility that ships with NVIDIA GPU drivers.

## Usage  

```Python
import nvidia_profiler

profiler = nvidia_profiler.Logger(fp_output_filename='test_name')

profiler.run_threaded()

program_under_test()

profiler.stop()
```

Data will be saved to the file path stored in `profiler.fname` and can be plotted with `nvidia_profiler.plot_data`. See the Jupyter Notebook `profile_example.ipynb` for a complete example.
