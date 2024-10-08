import logging
import gensim
import jieba
import BBA
import matplotlib.pyplot as plt
import torch

logging.basicConfig(level=logging.DEBUG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 载入停用词
stopwords = []
try:
    with open("/workspace/project/nlp_2024/word2vec/stopwords/hit_stopwords.txt", "r") as file:
        while True:
            line = file.readline()
            if not line:
                break
            stopwords.append(line.strip())
    with open("/workspace/project/nlp_2024/word2vec/stopwords/SpecialCharacters&PunctuationMarks.txt", "r") as file:
        while True:
            line = file.readline()
            if not line:
                break
            stopwords.append(line.strip('\n'))
except FileNotFoundError:
    logging.warning("No stopwords file found.")

# 中文训练
try:
    sentences = []
    with open("/workspace/project/nlp_2024/datasets/wiki/simplified_chinese_wiki.txt", "r") as file:  # 485228 simplified_chinese_wiki.txt
        read_count = int(485228 / 10000)
        line_count = 0
        while True:  # 使用一个无限循环
            line = file.readline()  # 读取一行
            if not line or line_count > read_count:  # 如果读取的行为空（即读到文件末尾），则跳出循环
                break
            sentences.append(list(jieba.cut(line.strip(), cut_all=False)))  # 输出该行并去掉尾部可能的换行符或空格
            line_count += 1
            if line_count % 10 == 0:
                logging.info(f'Loaded {line_count} | {read_count} .')
except FileNotFoundError:
    logging.warning("No train file found.")

# BBA CBOW
bbaCBOW = BBA.Word2Vec(sentences=sentences, stopwords=stopwords, vector_size=100, epochs=1, algorithm=0, min_count=1,
                       negative=5).to(device)

# BBA SkipGram
bbaSG = BBA.Word2Vec(sentences=sentences, stopwords=stopwords, vector_size=100, epochs=1, algorithm=1, min_count=1,
                     alpha=0.01, negative=5).to(device)

# 保存模型
bbaCBOW.save('/workspace/project/nlp_2024/word2vec/pth/bbaCBOW.pth')
bbaSG.save('/workspace/project/nlp_2024/word2vec/pth/bbaSG.pth')

# 加载模型
bbaCBOW = BBA.Word2Vec.load('/workspace/project/nlp_2024/word2vec/pth/bbaCBOW.pth').to(device)
bbaSG = BBA.Word2Vec.load('/workspace/project/nlp_2024/word2vec/pth/bbaSG.pth').to(device)

# CBOW：上下文词预测中心词
contexts = ["不丹的首都及最大城市为廷布",
            "无神论者不相信神存在，因为其缺乏经验证据的支持，无法解释罪恶问题",
            "亨利四世宣布成为英格兰国王",
            "物种起源解释了适应自然选择适者生存的原理",
            "资本主义天生与自由、平等和团结的价值观不相容",
            "国家石油公司的双峰塔曾经是世界最高建筑物",
            "南宁第一条铁路是湘桂铁",
            "虎跑泉水泡出的新茶尤为世人称道，因其色红香清如红梅，故称九曲红梅",
            "脑通常通过葡萄糖、血糖等的有氧代谢获得其大部分能量",
            "当时出身吴郡人严白虎等纠群结伙聚众数万，处处屯聚造反"]
for context in contexts:
    context_list = list(jieba.cut(context, cut_all=False))
    # word = context_list.pop(int(len(context_list) * 2 / 3))
    word = context_list.pop(-1)
    try:
        topn_words = bbaCBOW.most_similar(positive=context_list, topn=5)
        print(f'Context words: {context, word} -> Top {5} target words: {topn_words}')
    except KeyError:
        logging.warning(f'No words: {context_list} in vocab.')

# SkipGram：目标词预测其上下文词
targets = ["生理学", "服务", "操作系统", "战争", "化学家", "情境"]
for target in targets:
    try:
        topn_words = bbaSG.most_similar(positive=target, topn=5)
        print(f'Target word: {target} -> Top {5} context words: {topn_words}')
    except KeyError:
        logging.warning(f'No word: {target} in vocab.')

# 相似度预测
words = [
    # 包含
    ["操作系统", "内存",],
    ["戏剧", "歌唱",],
    ["文学", "小说",],
    ["气候", "寒冷",],

    # 关联
    ["乒乓球", "运动员"],
    ["三角形", "勾股定理"],
    ["武器", "海军"],
    ["电影", "首映"],

    # 反义词
    ["和平", "战争"],
    ["光明", "黑暗"],
    ["成功", "失败"],
    ["增加", "减少"],

    # 无关
    ["大桥", "包括"],
    ["欧盟", "方言"],
    ["分别", "公园"],
    ["地方", "陈独秀"],

]
for word1, word2 in words:
    try:
        cbowSimilarity = bbaCBOW.similarity(word1, word2)
        sgSimilarity = bbaSG.similarity(word1, word2)
        print(f'Similarity words: {word1, word2} -> cbow similarity: {cbowSimilarity}, sg similarity: {sgSimilarity}')
    except KeyError:
        logging.warning(f'No words: {word1,word2} in vocab.')

# 词向量可视化
words = ["衡量", "团体", "疫情", "问题", "领域", "社会", "困难", "压力", "差异", "尽管"]
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
try:
    cbow_reduced_embeddings = bbaCBOW.pca(words=words, n_components=2)
    sg_reduced_embeddings = bbaSG.pca(words=words, n_components=2)
    for i, word in enumerate(words):
        x1, y1 = cbow_reduced_embeddings[i]
        axs[0].scatter(x1, y1, label=i)
        axs[0].set_title('cbow_reduced_embeddings')
        axs[0].text(x1, y1, i, fontsize=12)

        x2, y2 = cbow_reduced_embeddings[i]
        axs[1].scatter(x2, y2, label=i)
        axs[1].set_title('sg_reduced_embeddings')
        axs[1].text(x2, y2, i, fontsize=12)
    axs[0].legend()
    axs[1].legend()
except KeyError:
    logging.warning(f'No words: {words} in vocab.')
plt.show()


# gensim测试
# dictionary = gensim.corpora.Dictionary(sentences)
# word2vec = gensim.models.Word2Vec(sentences=sentences, vector_size=100, epochs=10, sg=1, min_count=1)
# word2vec.save()

# target = '我们'
# topk_words = word2vec.wv.most_similar(target, topn=5)
# print(f'Target word: {target} -> Top {5} context words: {topn_words}')
# similarity = word2vec.wv.similarity('我们', '希望')
# print("The cosine similarity between 'word1' and 'word2' is: ", similarity)
