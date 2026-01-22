## 1. nvidia-smi查看gpu
    watch nvidia-smi
    
## 2. nvitop查看gpu
    pip install nvitop
    nvitop

## 3. 关掉所有使用gpu的程序
    fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh


