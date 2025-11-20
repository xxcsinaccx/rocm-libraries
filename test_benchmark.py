import subprocess
result = subprocess.run("mkdir -p /rocm-libraries/projects/rocprim/build", shell=True)
commands = [
    "ROCM_PATH=/opt/rocm CXX=hipcc cmake -DBUILD_BENCHMARK=ON -DBUILD_TEST=ON -DAMDGPU_TARGETS=gfx942 ../.",
    "make -j",
    "HIP_VISIBLE_DEVICES=7 ./benchmark/benchmark_warp_reduce --trials 20"
]
workdir = "/rocm-libraries/projects/rocprim/build"
for cmd in commands:
    print(f"running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=workdir)
    if result.returncode != 0:
        print(f"fail: {cmd}")
        break
