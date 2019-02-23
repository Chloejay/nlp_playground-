# coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#TensorFlow程序结构
#假设有两个向量 v_1 和 v_2 将作为输入提供给 Add 操作
#1. 定义图
v_1=tf.constant([1,2,3,4])
v_2=tf.constant([5,6,7,8])
v_add=tf.add(v_1,v_2)
#2. 会话中执行这个图
sess=tf.Session()
print(sess.run(v_add))

#********************
#张量、常量、变量、占位符
#张量:可理解为一个 n 维矩阵，所有类型的数据，包括标量、矢量和矩阵等都是特殊类型的张量
#常量：常量是其值不能改变的张量
#标量常量
c_1 = tf.constant(4)
#向量常量
c_2 = tf.constant([4,3,2])

#变量：当一个量在会话中的值需要更新时，使用变量来表示。在神经网络中，权重需要在训练期间更新，可以通过将权重声明为变量来实现，变量在使用前需要被显示初始化。
v1 = tf.Variable([1])
v2 = tf.Variable([1,2,3])

#占位符
#占位符不包含任何数据，因此不需要初始化它们，用于通常用于提供新的训练样本，feed_dict 一起使用来输入数据
v_1 = tf.placeholder(tf.int32, [4], name = 'v_1')
v_2 = tf.placeholder(tf.int32, [4], name = 'v_1')
v_add=tf.add(v_1,v_2)
result = sess.run(v_add, feed_dict = {v_1:[1,2,3,4],v_2:[5,6,7,8]})
print(result)

#基本操作
#定义矩阵

# 创建一个具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的截尾正态分布随机数
A = tf.truncated_normal([2,3])
print(sess.run(A))

# 矩阵元素全部是5的矩阵
B = tf.fill([2,3], 5.0)
print(sess.run(B))

# 创建一个具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的正态分布随机数
C = tf.random_uniform([3,2])
print(sess.run(C))
#随机
print(sess.run(C)) 
# 通过numpy 创建矩阵
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(D))
# 矩阵加减法
print(sess.run(A+B))
print(sess.run(B-B))
# 矩阵乘法
print(sess.run(tf.matmul(A,C )))
# 矩阵转置
print(sess.run(tf.transpose(C))) # Again, new random variables
# 矩阵行列式
print(sess.run(tf.matrix_determinant(D)))
# 矩阵的逆
print(sess.run(tf.matrix_inverse(D)))
# 矩阵的特征值和特征向量
print(sess.run(tf.self_adjoint_eig(D)))

#特殊函数
#数学函数略
#激活函数
x_vals = np.linspace(start=-10., stop=10., num=100)
# 整流线性单元（ Rectifier linear unit, Re LU ）是神经网络最常用的非线性函数。其函数为 max(O, ），连续但不平滑
print(sess.run(tf.nn.relu([-3., 3., 10.])))
y_relu = sess.run(tf.nn.relu(x_vals))
#sigmoid 函数是最常用的连续、平滑的激励函数 它也被称作逻辑函数（ Logistic数），表示为 l/(l+exp(-x）） sigmoid 函数由于在机器学习训练过程中反向传播项趋近 0,因此不怎么使用 使用方式如下：
print(sess.run(tf.nn.sigmoid([-1., 0., 1.])))
y_sigmoid = sess.run(tf.nn.sigmoid(x_vals))

plt.plot(x_vals, y_relu, 'b:', label='ReLU', linewidth=2)
plt.plot(x_vals, y_sigmoid, 'r--', label='Sigmoid', linewidth=2)
plt.ylim([-2,2])
plt.legend(loc='top left')
plt.show()

#特殊函数
#tf.nn.softmax_cross_entropy_with_logits 交叉熵函数 实现交叉熵的计算
y = np.array([[0, 10, 0], [0, 10, 0], [0, 10, 0], [0, 10, 0], [0, 10, 0]])
x = np.array([[0.0,10,0], [1.0,10,1], [5.0,10,5], [10,10,10], [20, 10,10]])
y = np.array(y).astype(np.float64)
result = sess.run(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x))
print(result)
#tf.reduce_mean  函数用于计算张量tensor沿着指定的数轴某一维度上的的平均值
x = tf.constant([[1., 2., 3.], [4., 5., 6.]])
mean1 = sess.run(tf.reduce_mean(x))
print(mean1)  
mean2 = sess.run(tf.reduce_mean(x, 0))
print(mean2)
mean3 = sess.run(tf.reduce_mean(x, 1))
print(mean3)
#tf.nn.embedding_lookup
data = np.array([[17,24,1],[23,5,7],[4,6,13],[10,12,19],[11,18,25]])
data = tf.convert_to_tensor(data)
lk = [3]#代表[0,0,0,1,0]
lookup_data = tf.nn.embedding_lookup(data,lk)
result = sess.run(lookup_data)
print(result)


