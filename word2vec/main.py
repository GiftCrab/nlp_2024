import logging
import gensim
import jieba
import BBA
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

# 载入停用词
stopwords = []
try:
    with open("stopwords/hit_stopwords.txt", "r") as file:
        while True:
            line = file.readline()
            if not line:
                break
            stopwords.append(line.strip())
    with open("stopwords/SpecialCharacters&PunctuationMarks.txt", "r") as file:
        while True:
            line = file.readline()
            if not line:
                break
            stopwords.append(line.strip())
except FileNotFoundError:
    logging.warning("No stopwords file found.")

# 中文训练
try:
    sentences = []
    with open("../datasets/wiki/simplified_chinese_wiki.txt", "r") as file:  # 使用"r"模式打开文件，表示read
        while True:  # 使用一个无限循环
            line = file.readline()  # 读取一行
            if not line:  # 如果读取的行为空（即读到文件末尾），则跳出循环
                break
            sentences.append(list(jieba.cut(line.strip(), cut_all=False)))  # 输出该行并去掉尾部可能的换行符或空格
except FileNotFoundError:
    logging.warning("No corpus file found.")

# BBA CBOW
bbaCBOW = BBA.Word2Vec(sentences=sentences, stopwords=stopwords, vector_size=100, epochs=10, algorithm=0, min_count=1,
                       negative=5)

# BBA SkipGram
bbaSG = BBA.Word2Vec(sentences=sentences, stopwords=stopwords, vector_size=100, epochs=10, algorithm=1, min_count=1,
                     negative=5)

# CBOW：上下文词预测中心词
contexts = ["中国便永远是这一样的中国，决不肯自己改变一根毫毛",
            "方桌上摆着十来碗饭菜",
            "正可给读他文章的所谓有教育的智识者嘻嘻一笑",
            "我于是仿佛看见雪白的桌布已经沾了许多酱油渍",
            "我仍然铺好被褥，用棉花裹了些他先前身体所在的地方的泥土",
            "路上是车夫们默默地前奔，似乎想赶紧逃出头上的烈日"]
for context in contexts:
    context_list = list(jieba.cut(context, cut_all=False))
    word = context_list.pop(int(len(context_list) * 2 / 3))
    topn_words = bbaCBOW.most_similar(positive=context_list, topn=5)
    print(f'Context words: {context, word} -> Top {5} target words: {topn_words}')

# SkipGram：目标词预测其上下文词
targets = ["天堂", "文学", "我们", "伟大", "革命", "医生"]
for target in targets:
    topn_words = bbaSG.most_similar(positive=target, topn=5)
    print(f'Target word: {target} -> Top {5} context words: {topn_words}')

# 相似度预测
words = [
    ["危险", "安全"],
    ["大家", "人民"],
    ["闲人", "如果"],
    ["天堂", "地域"],
    ["国家", "东欧"],
]
for word1, word2 in words:
    cbowSimilarity = bbaCBOW.similarity(word1, word2)
    sgSimilarity = bbaSG.similarity(word1, word2)
    print(f'Similarity words: {word1, word2} -> cbow similarity: {cbowSimilarity}, sg similarity: {sgSimilarity}')

# C词向量可视化
words = ["天堂", "文学", "我们", "伟大", "革命", "医生"]
cbow_reduced_embeddings = bbaCBOW.pca(words, n_components=2)
sg_reduced_embeddings = bbaSG.pca(words, n_components=2)
plt.figure(figsize=(10, 6))
for i, word_index in enumerate(words):
    x, y = cbow_reduced_embeddings[i]
    plt.scatter(x, y)
    plt.text(x, y, str(word_index), fontsize=9)
plt.title('bbaCBOW PCA of Word Embeddings')
plt.show()

# 保存模型
bbaCBOW.save('./pth/bbaCBOW.pth')
bbaSG.save('./pth/bbaSG.pth')

# 加载模型
bbaCBOW = BBA.Word2Vec.load('./pth/bbaCBOW.pth')
bbaCBOW.eval()
bbaSG = BBA.Word2Vec.load('./pth/bbaSG.pth')
bbaSG.eval()

# gensim测试
dictionary = gensim.corpora.Dictionary(sentences)
word2vec = gensim.models.Word2Vec(sentences=sentences, vector_size=100, epochs=10, sg=1, min_count=1)
word2vec.save()

target = '我们'
topk_words = word2vec.wv.most_similar(target, topn=5)
print(f'Target word: {target} -> Top {5} context words: {topn_words}')
similarity = word2vec.wv.similarity('我们', '希望')
print("The cosine similarity between 'word1' and 'word2' is: ", similarity)
