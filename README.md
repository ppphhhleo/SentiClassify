# SentiClassify
基于主题的情感分析，LSTM，aspect embedding，添加Attention机制

aspect-based sentiment
analysis 是文本分类的一个子任务

**输入**：一段文本； **输出**：主题-极性，比如：服务-积极（ positive
）、味道-消极（ negative ）、价格-中立（ neutral ）

<br />

---


## **QUICK START**

-   **文件**

    -   **/Data，测试集 训练集**

    -   **/Emb，已训练好的词向量，词典**

-   **main.py ，可选择模型** 

**LSTM，AELSTM（添加aspect embedding），ATLSTM（添加attention机制），ATAELSTM**  
</br>
选择完成后，运行main.py即开始训练模型



-   **model.py 定义以上模型**

 包括如何实现aspect 嵌入、注意力机制等 



-   **数据记录和作图，保存在record.ipynb文件中。**

---


<br />


### **0 数据集**

**SemEval2014 task4**

数据集主要用于细粒度情感分析，包含“Laptop”和“Restaurant”两个领域。

单条数据集，形式如：{**"sentence"**: "desserts are almost incredible: my
personal favorite is their tart of the day.", **"aspect"**: "food",
**"sentiment"**:
"positive"}，包含文本内容**sentence**、主题**aspect**和情感极性**sentiment**，其中，情感极性分为积极**positive、消极negative**和**中立neutral**三个水平。我们的目标是根据给定主题，来识别句子在该主题下包含的情感极性。

