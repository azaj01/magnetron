# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

mag_register_cpu_backend("loongarch64/mag_cpu_loongarch64_lsx.c" "-msimd=lsx" "")
mag_register_cpu_backend("loongarch64/mag_cpu_loongarch64_lasx.c" "-msimd=lasx" "") # todo: enable lasx
