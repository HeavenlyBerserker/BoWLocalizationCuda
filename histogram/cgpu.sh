salloc -n 1 -N 1 -t 1:00:00 -p soc-gpu-kp -A soc-gpu-kp --gres=gpu:p100:1
srun ./<executable>
