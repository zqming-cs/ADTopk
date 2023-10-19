bash /home/user/zwx/trans.sh ;
HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H n15:1,n16:1,n17:1,n18:1,n19:1,n20:1,n21:1,n22:1 python lstm.py --epochs 300
