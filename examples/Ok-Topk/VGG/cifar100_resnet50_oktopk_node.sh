#!/bin/bash -l
# SBATCH --nodes=2
# SBATCH --ntasks=2
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=1
# SBATCH --constraint=gpu
# SBATCH --partition=normal
# SBATCH --time=01:20:00
# SBATCH --output=16nodes_vgg_oktopk_density2.txt


# module load daint-gpu
source activate py39
which nvcc
nvidia-smi

which python


dnn="${dnn:-resnet50}"
density="${density:-0.001}"
source exp_configs/cifar100_resnet50.conf
# compressor="${compressor:-density}"

compressor="${compressor:-oktopk}"
nworkers="${nworkers:-8}"

max_epochs="${max_epochs:-80}"


echo $nworkers
nwpernode=1
sigmascale=2.5
PY=python




# srun $PY -m mpi4py main_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nwpernode $nwpernode --nsteps-update $nstepsupdate --compression --sigma-scale $sigmascale --density $density --compressor $compressor
# mpiexec -mca btl_tcp_if_include  ens39f1np1 -n $nworkers -host node16:1,node17:1 /home/user/anaconda3/envs/py39/bin/python  main_trainer_node.py  --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nwpernode $nwpernode --nsteps-update $nstepsupdate --compression --sigma-scale $sigmascale --density $density --compressor $compressor  


mpirun  -mca btl_tcp_if_include  ens39f1np1 -np $nworkers  -H node15:1,node16:1,node17:1,node18:1,node19:1,node20:1,node21:1,node22:1  /home/user/anaconda3/envs/py39/bin/python main_trainer_node_resnet.py  --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nwpernode $nwpernode --nsteps-update $nstepsupdate --compression --sigma-scale $sigmascale --density $density --compressor $compressor