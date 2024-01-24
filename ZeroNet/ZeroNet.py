import numpy as np
import matplotlib.pyplot as plt
import math
import time

from keras.datasets import mnist

np.random.seed(42)

Pi = math.pi
A = 24
B = 3.2
C = 42
D = 2.3
learning_rate = 0.02
epochs = 2000
batch_size = 100

train_loss = []
test_loss = []


# best
# SGD, lr=0.07, epochs=1000, batch_size=100, structure=(1,16), (16,16)x2, (16,1)
# Adam,lr=0.04, epochs=1000, batch_size=100, structure=(1,16), (16,16)x2, (16,1)
# Adam,lr=0.02, epochs=2000, batch_size=100, structure=(1,16), (16,16)x3, (16,1)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)))
    return labels_onehot


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


class ReLu(object):
    def __init__(self):
        super(ReLu, self).__init__()

    def __call__(self, x):
        self.x = x  # (N x dim)
        self.output = np.maximum(0, x)
        return self.output  # (N x dim)

    def backward(self, grad):
        """
        f(x) = ReLu(x)
        f'(x) = 1 if x > 0
        """
        grad[self.x <= 0] = 0  # ReLu函数的梯度
        return grad

    def step(self):
        """没有参数需要更新，所以pass"""
        pass


class Sigmoid(object):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def __call__(self, x):
        self.x = x  # (N x dim)
        self.output = 1 / (1 + np.exp(-self.x))
        return self.output  # (N x dim)

    def backward(self, grad):
        """
        f(x) = Sigmoid(x)
        f'(x) = f(x)(1-f(x))
        """
        self.sigmoid_grad = self.output * (1 - self.output)  # sigmoid函数的梯度
        return grad * self.sigmoid_grad

    def step(self):
        """没有参数需要更新，所以pass"""
        pass


class SGD(object):
    def __init__(self, lr=0.0001):
        super(SGD, self).__init__()
        self.lr = lr

    def __call__(self, grad):
        return -self.lr * grad


class Adam(object):
    def __init__(self, lr=0.0001, eps=1e-8, beta1=0.9, beta2=0.999):
        super(Adam, self).__init__()
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2

    def __call__(self, grad, m_t, v_t):
        m_t = self.beta1 * m_t + (1 - self.beta1) * grad
        v_t = self.beta2 * v_t + (1 - self.beta2) * (grad ** 2)

        m_hat = m_t / (1 - self.beta1)
        v_hat = v_t / (1 - self.beta2)

        return -self.lr * m_hat / (np.sqrt(v_hat) + self.eps), m_t, v_t


def kaiming_uniform(weights, bias=None, gain=math.sqrt(2)):
    """
    version: for ReLu
    梯度降不下去，weights和bias用kaiming初始化
    """
    fan_in = weights.shape[1]
    fan_out = weights.shape[0]
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    weights = np.random.uniform(-bound, bound, weights.shape)

    if bias is not None:
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        bias = np.random.uniform(-bound, bound, bias.shape)
        return weights, bias
    return weights


class Linear(object):
    def __init__(self, in_dim, out_dim, optimizer):
        super(Linear, self).__init__()
        self.optimizer = optimizer
        self.weights = np.zeros((in_dim, out_dim))  # 权重
        self.bias = np.zeros((out_dim,))  # 偏置

        # Adam优化器参数
        self.w_m_t = 0
        self.w_v_t = 0
        self.b_m_t = 0
        self.b_v_t = 0

        self.reset_parameters()  # 参数初始化

    def reset_parameters(self):
        if self.bias is not None:
            self.weights, self.bias = kaiming_uniform(self.weights, self.bias)
        else:
            self.weights = kaiming_uniform(self.weights)

    def __call__(self, x):
        self.input = x  # 输入
        self.output = np.dot(self.input, self.weights) + self.bias  # y = wx+b
        return self.output

    def backward(self, grad):
        N = self.input.shape[0]
        data_grad = np.dot(grad, self.weights.T)  # 当前层的梯度, dy/dx = W^T
        self.weights_grad = np.dot(self.input.T, grad) / N  # 当前层权重的梯度
        self.bias_grad = np.sum(grad, axis=0) / N  # 当前层偏置的梯度
        return data_grad

    def step(self):
        lr_weights = 0
        lr_bias = 0
        if self.optimizer.__class__.__name__ == 'SGD':
            lr_weights = self.optimizer(self.weights_grad)
            lr_bias = self.optimizer(self.bias_grad)
        elif self.optimizer.__class__.__name__ == 'Adam':
            lr_weights, self.w_m_t, self.w_v_t = self.optimizer(self.weights_grad, self.w_m_t, self.w_v_t)
            lr_bias, self.b_m_t, self.b_v_t = self.optimizer(self.bias_grad, self.b_m_t, self.b_v_t)
        self.weights += lr_weights
        self.bias += lr_bias


class MSELoss(object):
    def __init__(self):
        super(MSELoss, self).__init__()

    def __call__(self, y_predict, y_true):
        self.y_predict = y_predict
        self.y_true = y_true
        loss = np.mean(np.mean(np.square(self.y_predict - self.y_true), axis=-1))  # 损失函数值
        return loss

    def backward(self):
        grad = 2 * (self.y_predict - self.y_true) / (self.y_predict.shape[0] * self.y_predict.shape[1])  # 损失函数关于网络输出的梯度
        return grad

    def step(self):
        """没有参数需要更新，所以pass"""
        pass


class CrossEntropyLoss(object):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def __call__(self, y_predict, y_true):
        self.y_predict = y_predict
        y_exp = np.exp(self.y_predict)
        self.y_softmax = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
        # 避免数值溢出的等价式
        theta = self.y_predict - np.max(self.y_predict, axis=-1, keepdims=True)
        self.y_predict_logsoftmax = theta - np.log(np.sum(np.exp(theta), axis=-1, keepdims=True))
        self.y_true = y_true  # 必须是one-hot编码！
        loss = -np.sum(y_true * self.y_predict_logsoftmax) / self.y_true.shape[0]  # 损失函数值
        return loss

    def backward(self):
        grad = self.y_softmax - self.y_true / self.y_predict.shape[0]  # 损失函数关于网络输出的梯度
        return grad

    def step(self):
        """没有参数需要更新，所以pass"""
        pass


class ZeroNet(object):
    def __init__(self, optimizer=None):
        super(ZeroNet, self).__init__()
        self.network = [
            Linear(1, 16, optimizer=optimizer),
            ReLu(),
            Linear(16, 16, optimizer=optimizer),
            ReLu(),
            Linear(16, 16, optimizer=optimizer),
            ReLu(),
            Linear(16, 16, optimizer=optimizer),
            ReLu(),
            Linear(16, 1, optimizer=optimizer),
            # 分类用的softmax输出头靠CrossEntropyLoss完成
        ]

    def __call__(self, x):
        for layer in self.network:
            x = layer(x)
        return x

    def backward(self, grad):
        last_grad = grad.copy()
        for layer in self.network[::-1]:  # 倒序翻转
            last_grad = layer.backward(last_grad)
        return last_grad

    def step(self):
        for layer in self.network:
            layer.step()


def load_data(x_filename, y_filename):
    x_data = np.loadtxt(x_filename)
    x_data = x_data.reshape(x_data.shape[0], 1)
    y_data = np.loadtxt(y_filename)
    y_data = y_data.reshape(y_data.shape[0], 1)
    return x_data, y_data


class NumpyDataLoader(object):
    def __init__(self, x_train, y_train, batch_size=1, shuffle=True):
        super(NumpyDataLoader, self).__init__()
        self.x_train = x_train
        self.y_train = y_train

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.N = self.x_train.shape[0]
        self.indexes = [i for i in range(self.N)]

        self.queue = []
        self.ptr = 0

        if len(x_train.shape) > 2:  # 多维张量
            self.data_pretreat()

    def __call__(self):
        """便于每次迭代batch的时候直接取数据并再次随机shuffle"""
        self.data_shuffle()
        self.data_slice()
        return self.queue

    def data_pretreat(self):
        """将像素矩阵拉直成向量"""
        self.x_train.resize(self.N, 28 * 28)

    def data_shuffle(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

    def data_slice(self):
        self.queue = []  # 重新清理
        self.ptr = 0
        self.batch_num = self.N // self.batch_size
        for bn in range(self.batch_num):
            indexes_batch = self.indexes[self.ptr:self.ptr + self.batch_size]
            self.ptr += self.batch_size
            x_batch = self.x_train[indexes_batch]
            y_batch = self.y_train[indexes_batch]
            data_batch = [x_batch, y_batch]
            self.queue.append(data_batch)


def train(model, epochs, train_dataloader, x_test, y_test, criterion):
    for epoch in range(epochs):
        tmp_loss = 0
        for data in train_dataloader():
            x_train, y_train = data

            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            tmp_loss += loss.item()
            loss_grad = criterion.backward()
            print('epoch: {}, loss: {}'.format(epoch, loss.item()))
            # 梯度回传
            model.backward(loss_grad)
            # 梯度更新
            model.step()
        tmp_loss /= 10
        train_loss.append(tmp_loss)
        # testing
        test(x_test, y_test)


def test(x_test, y_test):
    y_pred = model(x_test)
    loss = criterion(y_pred, y_test)
    test_loss.append(loss.item())
    print('testing loss: {}'.format(loss.item()))


def train_mnist(model, epochs, train_dataloader, criterion):
    for epoch in range(epochs):
        for data in train_dataloader():
            x_train, y_train = data

            y_pred = model(x_train)
            acc = accuracy(y_pred, y_train)
            loss = criterion(y_pred, y_train)
            loss_grad = criterion.backward()
            print('epoch: {}, loss: {}, accuracy: {}'.format(epoch, loss.item(), acc))
            # 梯度回传
            model.backward(loss_grad)
            # 梯度更新
            model.step()


def accuracy(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)
    acc = np.mean(y_pred == y_true)
    return acc


if __name__ == '__main__':
    # train data
    x_train, y_train = load_data('data/x_train.csv', 'data/y_train.csv')
    train_dataloader = NumpyDataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)
    # test data
    x_test, y_test = load_data('data/x_test.csv', 'data/y_test.csv')
    # 模型初始化
    criterion = MSELoss()
    optimizer = Adam(lr=learning_rate)
    model = ZeroNet(optimizer=optimizer)
    # training
    start = time.time()
    train(model, epochs, train_dataloader, x_test, y_test, criterion)
    end = time.time()
    print('training finish! cost: {} s'.format(end - start))
    # testing
    y_pred = model(x_test)
    loss = criterion(y_pred, y_test)
    print('testing loss: {}'.format(loss.item()))
    # # 画loss图
    # plt.plot(range(2000), test_loss, label='test_loss', c='b')
    # plt.plot(range(2000), train_loss, label='train_loss', c='r')
    # plt.legend(loc='upper right')
    # plt.show()
    # # 画图
    # x = np.linspace(-2 * Pi, 2 * Pi, 1000)
    # y = [A * math.sin(B * i) + C * math.cos(D * i) for i in x]
    # y = np.array(y)
    # plt.plot(x, y, label='true value', c='blue')
    # plt.scatter(x_test, y_pred, label='predict value', c='red')
    # plt.legend(loc='upper right')
    # plt.show()
