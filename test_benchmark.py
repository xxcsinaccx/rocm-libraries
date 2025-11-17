import subprocess
result = subprocess.run("mkdir -p /rocm-libraries/projects/rocprim/build", shell=True)
commands = [
    "ROCM_PATH=/opt/rocm CXX=hipcc cmake -DBUILD_BENCHMARK=ON -DBUILD_TEST=ON -DAMDGPU_TARGETS=gfx942 ../.",
    "make -j",
    "./benchmark/benchmark_device_merge_sort --trials 20 --size 67108864"
]
workdir = "/rocm-libraries/projects/rocprim/build"
for cmd in commands:
    print(f"running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=workdir)
    if result.returncode != 0:
        print(f"fail: {cmd}")
        break
