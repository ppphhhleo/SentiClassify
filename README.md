# SentiClassify
基于主题的情感分析，LSTM，aspect embedding，添加Attention机制

aspect-based sentiment
analysis 是文本分类的一个子任务。文本分类是 NLP
的基础任务，旨在对给定文本预测其类别，应用场景广泛，比如垃圾邮件分类，微博情感分类，外卖评论，电影评论分类等。

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

