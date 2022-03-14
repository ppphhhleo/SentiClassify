from data_process import *
from model import *
import os
import torch.nn as nn
import time
#训练函数
def train(epochs):
	model.train() #模型设置成训练模式
	start_time = time.time()
	loss_list = []
	for epoch in range(epochs): #训练epochs轮
		loss_sum = 0  #记录每轮loss
		for batch in train_iter:
			input_, aspect, label = batch
			optimizer.zero_grad() #每次迭代前设置grad为0

			#不同的模型输入不同，请同学们看model.py文件
			# output = model(input_)  # LSTM
			output = model(input_, aspect) # AE / ATAE / AT

			loss = criterion(output, label) #计算loss
			loss.backward() #反向传播
			optimizer.step() #更新模型参数
			loss_sum += loss.item() #累积loss
		loss_list.append(loss_sum / len(train_iter))
		print('epoch: ', epoch, 'loss:', loss_sum / len(train_iter))
	end_time = time.time()
	print('time: {:5.3f}s'.format(end_time - start_time))
	print('loss = ', loss_list)
	test_acc = evaluate() #模型训练完后进行测试
	print('test_acc:', test_acc)

#测试函数
def evaluate():
	model.eval()
	total_acc, total_count = 0, 0
	loss_sum = 0

	with torch.no_grad(): #测试时不计算梯度
		for batch in test_iter:
			input_, aspect, label = batch

			# predicted_label = model(input_)  # LSTM
			predicted_label = model(input_, aspect) #  AE / ATAE / AT

			loss = criterion(predicted_label, label) #计算loss
			total_acc += (predicted_label.argmax(1) == label).sum().item() #累计正确预测数
			total_count += label.size(0) #累积总数
			loss_sum += loss.item() #累积loss
		print('test_loss:', loss_sum / len(test_iter))

	return total_acc/total_count


TORCH_SEED = 21 #随机数种子
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #设置模型在几号GPU上跑
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #设置device

# 设置随机数种子，保证结果一致
os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
np.random.seed(TORCH_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#创建数据集
train_dataset = MyDataset('./data/acsa_train.json')
test_dataset = MyDataset('./data/acsa_test.json')
train_iter = DataLoader(train_dataset, batch_size=25, shuffle=True, collate_fn=batch_process)
test_iter = DataLoader(test_dataset, batch_size=25, shuffle=False, collate_fn=batch_process)

# 加载我们的Embedding矩阵 已经训练好的，可以自行训练
embedding = torch.tensor(np.load('./emb/my_embeddings.npz')['embeddings'], dtype=torch.float)

#定义模型
# model = LSTM_Network(embedding).to(device)
# model = AELSTM_Network(embedding).to(device)
# model = ATAELSTM_Network(embedding).to(device)
model = ATLSTM_Network(embedding).to(device)



#定义loss函数、优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.001)

#开始训练
train(40)







