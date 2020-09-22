
#python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 30000 main.py --net LeNet --dataset MNIST --batch-size 128 --optimizer lars --lr 0.01,0.01 --grid_n 1 --lr_finder 1 --epochs 10 --wd 0.0001 --dist-url 'tcp://127.0.0.1:32345' &\

python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 30000 main.py --net LeNet --dataset MNIST --batch-size 32 --optimizer lars --lr 0.001,0.01 --warmup 5 --grid_n 10 --lr_finder 0 --epochs 10 --wd 0.0001 --dist-url 'tcp://127.0.0.1:32345' &\
python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 31000 main.py --net LeNet --dataset MNIST --batch-size 64 --optimizer lars --lr 0.001,0.01 --warmup 5 --grid_n 10 --lr_finder 0 --epochs 10 --wd 0.0001 --dist-url 'tcp://127.0.0.1:33345' &\
python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 32000 main.py --net LeNet --dataset MNIST --batch-size 128 --optimizer lars --lr 0.002,0.02 --warmup 5 --grid_n 10 --lr_finder 0 --epochs 10 --wd 0.0001 --dist-url 'tcp://127.0.0.1:34345' &\
python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 33000 main.py --net LeNet --dataset MNIST --batch-size 256 --optimizer lars --lr 0.002,0.02 --warmup 5 --grid_n 10 --lr_finder 0 --epochs 10 --wd 0.0001 --dist-url 'tcp://127.0.0.1:35345' &\
python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 34000 main.py --net LeNet --dataset MNIST --batch-size 512 --optimizer lars --lr 0.004,0.04 --warmup 5 --grid_n 10 --lr_finder 0 --epochs 10 --wd 0.0001 --dist-url 'tcp://127.0.0.1:36345' &\
python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 35000 main.py --net LeNet --dataset MNIST --batch-size 1024 --optimizer lars --lr 0.004,0.04 --warmup 5 --grid_n 10 --lr_finder 0 --epochs 10 --wd 0.0001 --dist-url 'tcp://127.0.0.1:37345' &\
#python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 36000 main.py --net LeNet --dataset MNIST --batch-size 2048 --optimizer lars --lr 0.01,0.1 --warmup 5 --grid_n 10 --lr_finder 0 --epochs 10 --wd 0.0001 --dist-url 'tcp://127.0.0.1:38345' &\
#python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 37000 main.py --net LeNet --dataset MNIST --batch-size 4096 --optimizer lars --lr 0.01,0.1 --warmup 5 --grid_n 10 --lr_finder 0 --epochs 10 --wd 0.0001 --dist-url 'tcp://127.0.0.1:39345' &\


echo "training mnist!"
