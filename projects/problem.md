I use `projects/rocprim/benchmark/benchmark_device_merge_sort.cpp` to test `cub::DeviceMergeSort::SortKeys` performance. But the performance is too bad (low bandwidth). To get a better bandwidth, you can focus on aligned and contiguous global memory access, shared memory usage with minimal bank conflicts, maximize GPU occupancy, reduce warp divergence, minimize global memory writes with efficient algorithm design. 

1. The files you need to view and those you can only edit are located in '/rocm-libraries/projects/rocprim/rocprim/include'.
3. The files in `/rocm-libraries/projects/rocprim/benchmark` is NOT allowed to be edited.
4. The file '/rocm-libraries/projects/rocprim/benchmark/benchmark_device_merge_sort.cpp' and '/rocm-libraries/projects/rocprim/benchmark/benchmark_device_merge_sort.hpp' are forbidden to be edited.
5. The file '/rocm-libraries/test_benchmark.py' is forbidden to be edited.
6. The file '/rocm-libraries/projects/rocprim/benchmark/benchmark_utils.hpp' is forbidden to be edited.
6. All CMAKEList and CMAKE files are forbidden to be edited.
7. you can use the follow commands to build and run benchmark in Docker:     
# running command
You can run `python /rocm-libraries/test_benchmark.py` to test the `cub::DeviceMergeSort::SortKeys` . Before run script test_benchmark.py, you need to read and check the code. After run `python /rocm-libraries/test_benchmark.py` you can get the bandwidth (bytes_per_second) of the kernel under different input key type and value type.