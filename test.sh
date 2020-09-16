#!/bin/sh



python3 test_on_mnist.py --batch-size 4196 --optimizer lars --epochs 10 --wd 0.0001 --lr 2,20 --g 0 &\
python3 test_on_mnist.py --batch-size 4196 --optimizer sgd --epochs 10 --wd 0.0001 --lr 0.05,1 --g 1 &\
python3 test_on_mnist.py --batch-size 4196 --optimizer adam --epochs 10 --wd 0.0001 --lr 0.005,0.1 --g 2 &\
python3 test_on_mnist.py --batch-size 2048 --optimizer lars --epochs 10 --wd 0.0001 --lr 2,20 --g 0 &\
python3 test_on_mnist.py --batch-size 2048 --optimizer sgd --epochs 10 --wd 0.0001 --lr 0.05,1 --g 1 &\
python3 test_on_mnist.py --batch-size 2048 --optimizer adam --epochs 10 --wd 0.0001 --lr 0.005,0.1 --g 2 &\
python3 test_on_mnist.py --batch-size 1024 --optimizer lars --epochs 10 --wd 0.0001 --lr 2,20 --g 0 &\
python3 test_on_mnist.py --batch-size 1024 --optimizer sgd --epochs 10 --wd 0.0001 --lr 0.05,1 --g 1 &\
python3 test_on_mnist.py --batch-size 1024 --optimizer adam --epochs 10 --wd 0.0001 --lr 0.005,0.1 --g 2 &\
#python3 test_on_mnist.py --batch-size 512 --optimizer lars --epochs 10 --wd 0.0001 --lr 2,20 --g 0 &\
#python3 test_on_mnist.py --batch-size 512 --optimizer sgd --epochs 10 --wd 0.0001 --lr 0.05,0.5 --g 1 &\
#python3 test_on_mnist.py --batch-size 512 --optimizer adam --epochs 10 --wd 0.0001 --lr 0.005,0.05 --g 2 &\
#python3 test_on_mnist.py --batch-size 128 --optimizer lars --epochs 10 --wd 0.0001 --lr 2,20 --g 0 &\
#python3 test_on_mnist.py --batch-size 128 --optimizer sgd --epochs 10 --wd 0.0001 --lr 0.05,0.5 --g 1 &\
#python3 test_on_mnist.py --batch-size 128 --optimizer adam --epochs 10 --wd 0.0001 --lr 0.005,0.05 --g 2 &\
#python3 test_on_mnist.py --batch-size 64 --optimizer lars --epochs 10 --wd 0.0001 --lr 1,10 --g 0 &\
#python3 test_on_mnist.py --batch-size 64 --optimizer sgd --epochs 10 --wd 0.0001 --lr 0.02,0.2 --g 1 &\
#python3 test_on_mnist.py --batch-size 64 --optimizer adam --epochs 10 --wd 0.0001 --lr 0.002,0.02 --g 2 &\
#python3 test_on_mnist.py --batch-size 32 --optimizer lars --epochs 10 --wd 0.0001 --lr 0.5,5 --g 0 &\
#python3 test_on_mnist.py --batch-size 32 --optimizer sgd --epochs 10 --wd 0.0001 --lr 0.01,0.1 --g 1 &\
#python3 test_on_mnist.py --batch-size 32 --optimizer adam --epochs 10 --wd 0.0001 --lr 0.001,0.01 --g 2
