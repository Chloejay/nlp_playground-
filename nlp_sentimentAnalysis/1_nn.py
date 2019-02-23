

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.framework import ops

#绘图显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#%matplotlib inline

#将 default graph 重新初始化，以保证内存中没有其他的 Graph
ops.reset_default_graph()
#加载 Iris 数据集
iris = datasets.load_iris()
#查看数据
iris.data
#存储花尊长度作为目标值
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])


#然后开始一个计算图会话
sess = tf.Session()

#设置一个种子使得返回结果可复现
seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)

# 准备数据集创建一个 80-20,分的训练集和测试集
#抽样
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

#也可以使用sklearn提供的划分训练集合测试集函数train_test_split
#x_vals_train,x_vals_test,y_vals_train,y_vals_test=model_selection.train_test_split(x_vals,y_vals,random_state=seed,test_size=0.2)

#通过 min-max归一化数据 0到1之间
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)
#0代替nan  
x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

#也可以使用sklearn提供的MinMaxScaler实现数据的归一化
#mm = MinMaxScaler()
#x_vals_train = np.nan_to_num(mm.fit_transform(x_vals_train))
#x_vals_test = np.nan_to_num(mm.fit_transform(x_vals_test))

#声明批量大小
batch_size = 50
#声明占位符
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#声明有合适形状的模型变量 我们能声明隐藏层为任意大小 ，本例中设置为有5个隐藏节点
hidden_layer_nodes = 10  #演示调整参数后看结果会有很大不同 5或者10
# 输入 ->隐藏层
A1 = tf.Variable(tf.random_normal(shape=[3,hidden_layer_nodes])) 
# 隐藏层节点的偏执
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
# 隐藏层 -> 输出层
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1])) 
# 输出层的偏执
b2 = tf.Variable(tf.random_normal(shape=[1]))   

#隐藏层输出
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
#模型的最后输出
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))
#这里定义均方误差作为损失函数
loss = tf.reduce_mean(tf.square(y_target - final_output))

#声明优化算法
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

#初始化模型变量
init = tf.initialize_all_variables()
sess.run(init)
#遍历迭代训练模型 我们也初始化两个列表（ ist ）存储训练损失和测试损失 在每次迭代训练时，随机选择批量训练数据来拟合模型

loss_vec = []
test_loss = []
for i in range(500):
    #首先，我们为每个批次随机生成一批索引
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    #通过索引值选择训练样本
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    #利用选择的数据进行训练
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    #计算并保存此次训练样本的损失
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(np.sqrt(temp_loss))
    #计算并保存整个测试集的损失
    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(np.sqrt(test_temp_loss))
    if((i+1)%50==0):
        print('第 ' + str(i+1) + '次迭代  训练样本损失=' + str(temp_loss) + "测试样本损失="+str(test_temp_loss))

#每次迭代的损失函数
plt.plot(loss_vec, 'k-', label='训练样本损失')
plt.plot(test_loss, 'r--', label='测试样本损失')
plt.title('每次迭代的损失')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.legend(loc='upper right')
plt.show()