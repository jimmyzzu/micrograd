import random
from micrograd.engine import Value

"""
这是所有网络组件的父类，模仿 PyTorch 中 nn.Module。
zero_grad()：把所有参数的梯度清零。
parameters()：返回模型所有参数列表，子类需要实现。
"""
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0  # 梯度归零

    def parameters(self):
        return []  # 默认无参数（子类重写）

"""
模拟一个神经元：
输入 x 是 List[Value]
输出是一个 Value（可自动微分）
如果 nonlin=True，则输出经过 ReLU 激活（即 max(0, x)）
"""
class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]  # nin 个输入的权重
        self.b = Value(0)                                           # 偏置项
        self.nonlin = nonlin                                        # 是否使用非线性激活（ReLU）

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)  # 线性组合 ∑ wi * xi + b
        return act.relu() if self.nonlin else act               # 激活函数（默认使用 ReLU）

    def parameters(self):
        return self.w + [self.b]  # 所有可学习参数：权重 + 偏置

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

"""
模拟一层全连接层：
每层由多个神经元组成
接收 nin 个输入，输出 nout 个输出（即神经元数量）
"""
class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]  # 生成 nout 个神经元

    def __call__(self, x):
        out = [n(x) for n in self.neurons]       # 每个神经元处理一次输入 x
        return out[0] if len(out) == 1 else out  # 如果只有 1 个神经元，直接返回它的输出

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]  # 所有神经元的参数

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

"""
模拟一个多层神经网络：
参数 nouts 是一个 list，定义每层的神经元数量
自动构造若干层 Layer
除最后一层外，默认都使用 ReLU 非线性
"""
class MLP(Module):
    """
    Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1)
    这行表示：
        ● 构建一个 Layer，输入维度是 sz[i]，输出维度是 sz[i+1]
        ● 是否使用非线性激活（ReLU）由 nonlin=... 决定：nonlin=i != len(nouts) - 1
        ● 对除了最后一层以外的所有层，都设置 nonlin=True（使用 ReLU）
        ● 对最后一层，nonlin=False（不使用 ReLU），因为一般输出层不用激活或用 softmax、sigmoid 等其他函数
        
    mlp = MLP(3, [4, 4, 1])
    就会构建：
    self.layers = [
        Layer(3, 4, nonlin=True),   # 隐藏层 1
        Layer(4, 4, nonlin=True),   # 隐藏层 2
        Layer(4, 1, nonlin=False)   # 输出层（没有 ReLU）
    ]
    """
    def __init__(self, nin, nouts):
        sz = [nin] + nouts  # 层大小 [输入维度, 隐藏层1, 隐藏层2, ..., 输出层], 这是将 输入维度 nin 和 每层输出维度 nouts 合并成一个列表，表示每一层的输入输出大小。
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i !=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)  # 层层调用__call__方法
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]  # 所有层的参数

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
