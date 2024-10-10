import logging
import torch
import numpy
from collections import Counter, defaultdict
import utils
import datetime
from SkipGram import SkipGramModel, SkipGramDataset
from CBOW import CBOWModel, CBOWDataset
from sklearn.decomposition import PCA
import gc


class Word2Vec(torch.nn.Module):
    def __init__(
            self, sentences=None, stopwords=None, vector_size=100, window=5, min_count=5, workers=0, algorithm=0,
            hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, alpha=0.01, min_alpha=0.001,
            epochs=5, compute_loss=False, seed=1, max_final_vocab=None, sample=1e-3, sorted_vocab=True,
            batch_words=256, trim_rule=None, callbacks=(), shrink_windows=True
    ):
        """
        Baby Broca's Area.

        TODO 介绍

        参数:
            sentences (iterable， 必选): 训练数据，可以是一个列表，包含多个句子，每个句子是一个词的列表。也可以是一个可迭代对象。默认为 None。
            stopwords (iterable， 可选): 停用词列表。默认为 None。
            vector_size (int, 可选): 生成的词向量的维度。默认为 100。
            window (int, 可选): 上下文窗口的大小，表示当前词语前后可以看多远的词。默认为 5。
            min_count (int, 可选): 忽略所有频率低于此值的词。默认为 5。
            workers (int, 可选): 并行训练时使用的线程数。默认为 1。
            algorithm (int, 可选): 训练算法的选择，0 表示 CBOW，1 表示 Skip-gram。 默认为 0。
            # hs (int, 可选): 如果为 1，则使用层次 Softmax；如果为 0，则使用负采样。 默认为 0。
            negative (int, 可选): 如果 > 0，则使用负采样，每个正采样对应的负采样的个数。默认为 5。
            ns_exponent (float, 可选): 负采样分布的指数值。1.0 表示与 unigram 分布相同，0.0 表示所有词的频率相同。默认为  0.75。
            cbow_mean (int, 可选): 如果为 1，则使用上下文词向量的均值；如果为 0，则使用总和。仅在 CBOW 模式下使用。默认为 1。
            alpha (float, 可选): 初始学习率。默认为 0.025。
            min_alpha (float, 可选): 训练过程中线性衰减到的最小学习率。默认为 0.0001。
            epochs (int, 可选): 训练语料库的迭代次数。默认为 5。
            compute_loss (bool, 可选): 如果为 True，则计算并存储训练期间的损失值。默认为 False。
            seed (int, 可选): 随机数生成器的种子。默认为 1。
            sample (float, 可选): 高频词的下采样阈值。默认为 1e-3。
            sorted_vocab (bool, 可选): 如果为 True，则在构建词汇表时按词频排序。默认为 True。
            batch_words (int, 可选): 每个工作线程的批处理大小。默认为 256。
            trim_rule (func, 可选): 指定自定义词汇表修剪规则的函数。默认为 None.
            callbacks (list of func, 可选): 在训练过程中调用的回调函数列表。默认为 None.
            shrink_windows (bool, 可选): 如果为 True，则在训练过程中逐渐减小窗口大小。默认为 False。
        """

        super(Word2Vec, self).__init__()
        if sentences is None:
            return
        # self.sentences = sentences
        self.stopwords = stopwords or []
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.algorithm = algorithm
        # self.hs = hs
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_mean = cbow_mean
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.epochs = epochs or 5
        self.compute_loss = compute_loss
        self.seed = seed
        self.sample = sample
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.trim_rule = trim_rule
        self.callbacks = callbacks
        self.shrink_windows = shrink_windows

        # 运行中参数
        self.raw_vocab_freq = None  # 原始词频
        self.corpus_count = None  # 语料库条数
        self.corpus_total_words = None  # 语料库总词数
        self.vocab_size = None  # 词汇表大小
        self.index_to_key = [None]  # idx_to_word
        self.key_to_index = {}  # word_to_idx
        self.context_vectors = None  # 上下文词向量
        self.context_norms = None  # 上下文词向量模
        self.target_vectors = None  # 目标词向量
        self.target_norms = None  # 目标词向量模
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 数据预处理
        tokens = self.prepare_data(sentences=sentences, stopwords=self.stopwords, )

        # 2. 词汇表构建
        self.build_vocab(sentences=tokens, min_count=self.min_count,
                         sorted_vocab=self.sorted_vocab,
                         negative=self.negative, ns_exponent=self.ns_exponent)

        # 3. 训练词向量
        self.train(sentences=tokens, epochs=self.epochs,
                   algorithm=self.algorithm,
                   window=self.window,
                   key_to_index=self.key_to_index,
                   vocab_size=self.vocab_size,
                   negative=self.negative,
                   word_freq_dist=self.word_freq_dist,
                   batch_words=self.batch_words,
                   workers=self.workers,
                   vector_size=self.vector_size,
                   start_alpha=self.alpha, end_alpha=self.min_alpha,
                   compute_loss=self.compute_loss)

    def prepare_data(self, sentences=[], stopwords=[]):
        """
        数据预处理：去除停用词
        """
        logging.debug(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - prepare_data')
        tokens = []
        for sentence in sentences:
            token = []
            for word in sentence:
                # 去除停用词
                if word not in stopwords:
                    token.append(word)
            if len(token) > 0:
                tokens.append(token)
        # TODO 优化sentences和tokens的区别
        # self.sentences = tokens
        return tokens

    def build_vocab(
            self, sentences=None, min_count=5, sorted_vocab=True, negative=None, ns_exponent=0.75,
            **kwargs,
    ):
        """
        根据sentences 构建词汇表
        词频、语料库、总词数、词汇表大小、min_count、词频排序、key和index映射、负采样表
        """
        logging.debug(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - build_vocab')
        assert sentences is not None

        # word_freq = Counter(list(itertools.chain.from_iterable(sentences)))

        # 使用节省内存的方式
        total_words = 0
        word_freq = defaultdict(int)
        for sentence in sentences:
            for word in sentence:
                word_freq[word] += 1
            total_words += len(sentence)

        # 原始词频
        self.raw_vocab_freq = word_freq
        # 语料库条数：len[sentences]
        self.corpus_count = len(sentences)
        # 语料库总词数
        self.corpus_total_words = total_words

        # 丢弃频率低于 min_count 的单词
        for key in list(word_freq.keys()):
            if word_freq[key] < min_count:
                del word_freq[key]

        gc.collect()  # 手动调用垃圾回收器

        # 词频排序，设置index和key映射
        if sorted_vocab:  # 排序只改变index_to_key和key_to_index，不改变词频raw_vocab_freq
            raw_vocab_freq_sorted = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
            self.index_to_key = [word for word, freq in raw_vocab_freq_sorted]
        else:
            self.index_to_key = [word for word, freq in word_freq.items()]

        # 设置key和index映射
        self.key_to_index = {word: i for i, word in enumerate(self.index_to_key)}
        self.vocab_size = len(self.index_to_key)

        if negative:
            # 构建用于负采样的抽取随机词的表格
            self.make_cum_table(index_to_key=self.index_to_key, raw_vocab_freq=self.raw_vocab_freq,
                                vocab_size=self.vocab_size, ns_exponent=ns_exponent,
                                domain=2 ** 31 - 1)

            # 负样本：生成负样本的概率分布
            word_freq_array = numpy.array([self.raw_vocab_freq[word] for word in self.raw_vocab_freq])
            # 对词频数组进行幂运算，通常用于调整词频分布，减少高频词的影响。这是 Word2Vec 中常用的一种技巧。
            word_freq_array = word_freq_array ** ns_exponent
            # 将词频数组归一化，使其总和为1，从而形成一个有效的概率分布。
            word_freq_array /= numpy.sum(word_freq_array)
            self.word_freq_dist = torch.distributions.Categorical(
                probs=torch.tensor(word_freq_array, dtype=torch.float32))

    def make_cum_table(self, index_to_key={}, raw_vocab_freq=None, vocab_size=None, ns_exponent=0.75,
                       domain=2 ** 31 - 1):
        """
        构建一个累积分布表（cumulative distribution table），用于在负采样中高效地选择单词。
        基于词频的次幂分布，并且被归一化到一个指定的域domain（默认为2 ** 31 - 1）
        """

        self.cum_table = numpy.zeros(vocab_size, dtype=numpy.uint32)

        # 计算所有功率的总和（论文中的 Z）
        train_words_pow = 0.0
        for word_index in range(vocab_size):
            freq = raw_vocab_freq[index_to_key[word_index]]
            train_words_pow += freq ** float(ns_exponent)

        cumulative = 0.0
        for word_index in range(vocab_size):
            freq = raw_vocab_freq[index_to_key[word_index]]
            cumulative += freq ** float(ns_exponent)
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def train(
            self, sentences=None,
            algorithm=0,
            epochs=5, window=5,
            key_to_index={},
            vocab_size=None,
            negative=5,
            word_freq_dist=None,
            batch_words=256,
            workers=0,
            vector_size=100,
            start_alpha=0.025, end_alpha=0.001,
            compute_loss=False,
            **kwargs,
    ):
        """
        训练词向量

        Parameters
        ----------
        TODO 参数描述

        """

        logging.debug(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - train')

        assert sentences is not None

        # TODO if compute_loss : running_training_loss = 0.0

        # 准备数据
        # 定义模型
        if algorithm == 0:  # 使用CBOW模型
            dataset = CBOWDataset(
                sentences=sentences, window_size=window, word_to_idx=key_to_index,
                vocab_size=vocab_size, num_neg_samples=negative)
            model = CBOWModel(vocab_size=vocab_size, embedding_dim=vector_size, ).to(self.device)
        else:  # 使用SkipGram模型
            dataset = SkipGramDataset(
                sentences=sentences, window_size=window, word_to_idx=key_to_index,
                vocab_size=vocab_size, num_neg_samples=negative)
            model = SkipGramModel(vocab_size=vocab_size, embedding_dim=vector_size, ).to(self.device)  # initial_embeddings=self.vectors

        # TODO num_workers workers
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_words, shuffle=True, num_workers=workers)

        # 设置优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=start_alpha)

        # 学习率更新函数
        def lr_lambda(epoch): return 1.0 if epochs == 1 else 1 - (1 - end_alpha / start_alpha) * epoch / (epochs - 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        # 训练
        lossmap = {}
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            train_epoch = 0
            for data, target, n in train_loader:
                positive_data, positive_target = data.to(self.device), target.to(self.device)

                if negative > 0 and word_freq_dist is not None:
                    negative_data = torch.cat([word_freq_dist.sample((positive_data.shape)) for _ in range(negative)]).to(self.device)
                    negative_target = torch.cat([positive_target for _ in range(negative)]).to(self.device)

                optimizer.zero_grad()
                loss = model(positive_context=positive_data, positive_target=positive_target,
                             negative_context=negative_data, negative_target=negative_target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_epoch += 1

                if train_epoch % 100 == 0:
                    logging.debug(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {type(model).__name__} - Epoch [{epoch + 1}/{epochs}], Training: {train_epoch*batch_words}/{len(dataset)}')

            if compute_loss:
                lossmap[epoch + 1] = epoch_loss / len(train_loader)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logging.debug(
                f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {type(model).__name__} - Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.10f}, LearningRate: {optimizer.param_groups[0]["lr"]:.5f}')

            scheduler.step()

        model.eval()  # 训练完成

        # 训练结束：更新词嵌入表
        self.context_vectors = model.context_embeddings.weight.detach().cpu().numpy()
        self.target_vectors = model.target_embeddings.weight.detach().cpu().numpy()
        self.context_norms = numpy.linalg.norm(self.context_vectors, axis=1)
        self.target_norms = numpy.linalg.norm(self.target_vectors, axis=1)
        self.lossmap = lossmap

    def most_similar(
            self, positive=None, negative=None, topn=10, decimal=8, clip_start=0, clip_end=None,
            restrict_vocab=None,
    ):
        """
        查找前 N 个最相似的词
        positive相似度有正向贡献，negative有负向贡献。
        此方法计算给定键的投影权重向量的简单平均值与模型中每个键的向量之间的余弦相似度。

        Parameters
        ----------
        positive : 可选，列表类型，元素为字符串、整数或ndarray，或者是元组（元组的第一个元素为字符串、整数或ndarray，第二个元素为权重，默认值为1.0）列表中的键会产生正面的贡献。
        negative : 可选，列表类型，元素为字符串、整数或ndarray，或者是元组（元组的第一个元素为字符串、整数或ndarray，第二个元素为权重，默认值为-1.0）列表中的键会产生负面的贡献。
        topn : 可选，整数或None。如果topn为整数，函数返回与键最相似的前N个键；如果topn为None，则返回所有键的相似度。
        clip_start : 整数。剪辑的起始索引。
        clip_end : 整数。剪辑的结束索引。
        restrict_vocab : 可选，整数。限制在寻找最相似值时检查的向量范围的可选整数参数。例如，restrict_vocab=10000只会检查词汇顺序中的前10000个键向量。（如果你按降序频率排序了词汇表，这可能有意义。）如果指定了此参数，它会覆盖clip_start或clip_end的任何值。

        TODO restrict_vocab
        restrict_vocab是一个整数，用于限制语意空间的大小。对于巨大的模型，如果你只需要查找最常用词，那么这个参数就非常有用。例如，restrict_vocab=50000会只在最常见的50000个词中寻找最相似的词。
        clip_end也是一个整数，它是most_similar方法返回的相似单词数量的上限。如果clip_end=10，那么无论多少词的相似度都超过了阈值，都只会返回相似度最高的前10个词。
        restrict_vocab限制了模型在寻找相似词时所搜索的词汇空间数量，而clip_end限制了模型返回的相似词的数量。


        返回
        -------
        (str, float) 列表或 numpy.array
            当 `topn` 为 int 时，将返回 (key, similarity) 序列。
            当 `topn` 为 None 时，将返回所有键的相似度作为
            具有词汇表大小的一维 numpy 数组。
        """

        if topn < 1:
            return []

        positive = self._ensure_list(positive)
        negative = self._ensure_list(negative)

        clip_end = clip_end or self.vocab_size

        if restrict_vocab:
            clip_start = 0
            clip_end = restrict_vocab

        # 如果尚未存在，则为每个键添加权重；对于正键，默认为 1.0；对于负键，默认为 -1.0
        keys = []
        weight = numpy.concatenate((numpy.ones(len(positive)), -1.0 * numpy.ones(len(negative))))
        for idx, item in enumerate(positive + negative):
            if isinstance(item, (str, int, numpy.integer, numpy.ndarray)):
                keys.append(item)
            else:
                keys.append(item[0])
                weight[idx] = item[1]

        # compute the weighted average of all keys
        mean = self.get_mean_vector(keys, weight, vectors=self.context_vectors, norms=self.context_norms,
                                    pre_normalize=True, post_normalize=True,
                                    ignore_missing=False)
        all_keys = [
            self.key_to_index[key] for key in keys if self.exist_key(key)
        ]

        # 计算每个向量与 mean 向量的点积
        dists = numpy.dot(
            [self.get_vector_by_index(index=i, vectors=self.target_vectors, norms=self.target_norms, norm=True) for i
             in range(clip_start, clip_end)], mean/numpy.linalg.norm(mean))
        if not topn:
            return dists

        best = utils.argsort(dists, topn=topn + len(all_keys), reverse=True)
        # 忽略（不返回）输入中的键
        result = [
            (self.index_to_key[sim + clip_start], round(float(dists[sim]), decimal))
            for sim in best if (sim + clip_start) not in all_keys
        ]
        return result[:topn]

    def _ensure_list(self, value):
        """
        确保将指定的值包装在列表中，以适应那些我们也接受单个键或向量的情况。
        """
        if value is None:
            return []

        if isinstance(value, (str, int, numpy.integer)) or (isinstance(value, numpy.ndarray) and len(value.shape) == 1):
            return [value]

        if isinstance(value, numpy.ndarray) and len(value.shape) == 2:
            return list(value)

        return value

    def get_mean_vector(self, keys, weights=None, vectors=None, norms=None, pre_normalize=True, post_normalize=False,
                        ignore_missing=True):
        """
        获取给定键列表的平均向量。

        Parameters
        ----------
        TODO 参数描述

        """
        if len(keys) == 0:
            raise ValueError("cannot compute mean with no input")
        if isinstance(weights, list):
            weights = numpy.array(weights)
        if weights is None:
            weights = numpy.ones(len(keys))
        if len(keys) != weights.shape[0]:  # weights is a 1-D numpy array
            raise ValueError(
                "keys and weights array must have same number of elements"
            )

        mean = numpy.zeros(self.vector_size, vectors.dtype)

        total_weight = 0
        for idx, key in enumerate(keys):
            if self.exist_key(key):
                vec = self.get_vector_by_key(key=key, key_to_index=self.key_to_index, vectors=vectors, norms=norms,
                                             norm=pre_normalize)
                mean += weights[idx] * vec
                total_weight += weights[idx]
            elif not ignore_missing:
                raise KeyError(f"Key '{key}' not present in vocabulary")

        if total_weight > 0:
            mean = mean / numpy.abs(total_weight)
        return mean

    def get_vector_by_key(self, key, key_to_index, vectors, norms, norm=False):
        """
        获取键的向量，作为 1D numpy 数组。

        Parameters
        ----------
        TODO 参数描述

        """
        index = key_to_index.get(key, -1)
        if norm:
            # self.fill_norms()
            result = vectors[index] / norms[index]
        else:
            result = vectors[index]

        result.setflags(write=False)  # disallow direct tampering that would invalidate `norms` etc
        return result

    def get_vector_by_index(self, index, vectors=None, norms=None, norm=False):
        """
        获取键的向量，作为 1D numpy 数组。

        Parameters
        ----------
        TODO 参数描述

        """
        if norm:
            result = vectors[index] / norms[index]
        else:
            result = vectors[index]

        result.setflags(write=False)  # disallow direct tampering that would invalidate `norms` etc
        return result

    def exist_key(self, key):
        """
        检查给定的键是否存在于词汇表中
        """
        return self.key_to_index.get(key, -1) >= 0

    def similarity(self, w1, w2):
        """
        计算两个键之间的余弦相似度

        Parameters
        ----------
        w1 : str
            单词1.
        w2 : str
            单词2.

        Returns
        -------
        float
            `w1` 和 `w2` 之间的余弦相似度。

        """
        return numpy.dot(self.unit_vec(self.key_to_index[w1], self.target_vectors, self.target_norms, return_norm=True),
                         self.unit_vec(self.key_to_index[w2], self.target_vectors, self.target_norms, return_norm=True))

    def unit_vec(self, index, vectors, norm, return_norm=False):
        """
        将矢量缩放到单位长度

        Parameters
        ----------
        TODO 参数描述

        vec : {numpy.ndarray, scipy.sparse, list of (int, float)}
            Input vector in any format
        norm : {'l1', 'l2', 'unique'}, optional
            Metric to normalize in.
        return_norm : bool, optional
            Return the length of vector `vec`, in addition to the normalized vector itself?

        Returns
        -------
        numpy.ndarray, scipy.sparse, list of (int, float)}
            Normalized vector in same format as `vec`.
        float
            Length of `vec` before normalization, if `return_norm` is set.

        Notes
        -----
        Zero-vector will be unchanged.

        """
        # norm = 'l2'
        # veclen = numpy.sqrt(np.sum(vec.data ** 2))
        if return_norm:
            return vectors[index] / norm[index]
        else:
            return vectors[index]

    def pca(self, words=None, n_components=2):
        """
        主成分分析
        """
        _words = self._ensure_list(words)
        words_embeddings = [self.target_vectors[self.key_to_index[word]] if word in self.key_to_index else numpy.zeros(self.vector_size) for word in _words]
        pca = PCA(n_components=n_components)
        return pca.fit_transform(words_embeddings)

    def save(self, filename):
        # TODO  save权重和属性
        torch.save({
            'stopwords': self.stopwords,
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'workers': self.workers,
            'algorithm': self.algorithm,
            # self.hs = hs
            'negative': self.negative,
            'ns_exponent': self.ns_exponent,
            'cbow_mean': self.cbow_mean,
            'alpha': self.alpha,
            'min_alpha': self.min_alpha,
            'epochs': self.epochs,
            'compute_loss': self.compute_loss,
            'seed': self.seed,
            'sample': self.sample,
            'sorted_vocab': self.sorted_vocab,
            'batch_words': self.batch_words,
            'trim_rule': self.trim_rule,
            'callbacks': self.callbacks,
            'shrink_windows': self.shrink_windows,

            # 运行中参数
            'raw_vocab_freq': self.raw_vocab_freq,
            'corpus_count': self.corpus_count,
            'corpus_total_words': self.corpus_total_words,
            'vocab_size': self.vocab_size,
            'index_to_key': self.index_to_key,
            'key_to_index': self.key_to_index,
            'context_vectors': self.context_vectors,
            'context_norms': self.context_norms,
            'target_vectors': self.target_vectors,
            'target_norms': self.target_norms,
            'device': self.device,

            'model_state_dict': self.state_dict(),
        }, filename)

    @classmethod
    def load(self, filename):
        # load权重和属性
        checkpoint = torch.load(filename)
        # model = self(checkpoint['name'], checkpoint['value'])
        word2vec = self()
        word2vec.stopwords = checkpoint['stopwords']
        word2vec.vector_size = checkpoint['vector_size']
        word2vec.window = checkpoint['window']
        word2vec.min_count = checkpoint['min_count']
        word2vec.workers = checkpoint['workers']
        word2vec.algorithm = checkpoint['algorithm']
        # word2vec.hs = checkpoint['hs']
        word2vec.negative = checkpoint['negative']
        word2vec.ns_exponent = checkpoint['ns_exponent']
        word2vec.cbow_mean = checkpoint['cbow_mean']
        word2vec.alpha = checkpoint['alpha']
        word2vec.min_alpha = checkpoint['min_alpha']
        word2vec.epochs = checkpoint['epochs']
        word2vec.compute_loss = checkpoint['compute_loss']
        word2vec.seed = checkpoint['seed']
        word2vec.sample = checkpoint['sample']
        word2vec.sorted_vocab = checkpoint['sorted_vocab']
        word2vec.batch_words = checkpoint['batch_words']
        word2vec.trim_rule = checkpoint['trim_rule']
        word2vec.callbacks = checkpoint['callbacks']
        word2vec.shrink_windows = checkpoint['shrink_windows']

        word2vec.raw_vocab_freq = checkpoint['raw_vocab_freq']
        word2vec.corpus_count = checkpoint['corpus_count']
        word2vec.corpus_total_words = checkpoint['corpus_total_words']
        word2vec.vocab_size = checkpoint['vocab_size']
        word2vec.index_to_key = checkpoint['index_to_key']
        word2vec.key_to_index = checkpoint['key_to_index']
        word2vec.context_vectors = checkpoint['context_vectors']
        word2vec.context_norms = checkpoint['context_norms']
        word2vec.target_vectors = checkpoint['target_vectors']
        word2vec.target_norms = checkpoint['target_norms']
        word2vec.device = checkpoint['device']
        word2vec.load_state_dict(checkpoint['model_state_dict'])
        return word2vec
