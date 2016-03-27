# Deep-Residual-Network
Implementation of deep residual networks with inception bottleneck in Lasagne

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Identity Mappings in Deep Residual Networks", http://arxiv.org/abs/1603.05027

Implemented various bottleneck architecture, dropout bottleneck etc
Proposed new bootleneck architecture using modified Inception module

## Usage
 
 THEANO_FLAGS=device=gpu0,floatX=float32,profile=True python resnetcifar.py [options]
