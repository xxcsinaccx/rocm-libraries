import subprocess

commands = [
    "ROCM_PATH=/opt/rocm CXX=hipcc cmake -DBUILD_BENCHMARK=ON -DAMDGPU_TARGETS=gfx942 ../.",
    "make -j benchmark_device_merge_sort",
    "make install",
    "./benchmark/benchmark_device_merge_sort --trials 20 --size 67108864"
]
workdir = "/rocm-libraries/projects/rocprim/build"
for cmd in commands:
    print(f"running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=workdir)
    if result.returncode != 0:
        print(f"fail: {cmd}")
        break
