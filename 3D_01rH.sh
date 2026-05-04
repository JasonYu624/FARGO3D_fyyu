#!/bin/bash
#SBATCH --job-name=3D_01rH
#SBATCH --account=def-evelee
#SBATCH --time=00:30:00

#SBATCH --nodes=1                 # 只用 1 个节点
#SBATCH --ntasks-per-node=4       # 4 个 MPI 进程
#SBATCH --cpus-per-task=1         # 每个进程 8 个 CPU 核（适当给多一点给 I/O、重力等 CPU 部分）
#SBATCH --gpus-per-node=h100:4    # 这个节点上要 4 张 H100
#SBATCH --mem-per-cpu=20G                 # 预留足够内存（你之前只给了 16G，3D 会容易不够）


# 推荐：打印一点环境信息，方便 debug
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOB_ID"

# 加载你需要的模块
module purge
module load StdEnv/2023
module load cuda/12.6
module load openmpi/4.1.5
# 切到你的 fargo3d 目录（根据实际路径修改）
cd /home/fyyu/projects/rrg-rbdong/fyyu/fargo3d_modify

export OMP_NUM_THREADS=1
export OMPI_MCA_opal_cuda_support=true
srun nvidia-smi

# mpirun -np 4 ./setups/3D_01rH_gas_vcap/fargo3d setups/3D_01rH_gas_vcap/3D_01rH_gas.par > run.log 2>&1
# grep -i -E "error|cuda|illegal|nan|failed|abort|segmentation|memory" run.log
mpirun -np 4 ./setups/3D_01rH_gas_vcap_accret/fargo3d setups/3D_01rH_gas_vcap_accret/3D_01rH_gas_vcap_accret.par

# 确保写盘完成（一般不必，但有时并行文件系统上更稳）
sync

# 删除（-f：没有匹配也不报错）
rm -f outputs/3D_01rH_vcap/*_2d.dat

# 校验：如果还有残留就直接报错退出
if ls outputs/3D_01rH_vcap/*_2d.dat >/dev/null 2>&1; then
  echo "ERROR: still exists outputs/3D_01rH/*_2d.dat after rm"
  ls -l outputs/3D_01rH_vcap/*_2d.dat || true
  exit 1
fi

mpirun -np 4 ./setups/3D_01rH_vcap_accret/fargo3d -p -S 20 setups/3D_01rH_vcap_accret/3D_01rH_vcap_accret.par
# mpirun -np 4 ./setups/3D_01rH/fargo3d -p -S 262 setups/3D_01rH/3D_01rH.par
echo "Job finished on $(date)"
