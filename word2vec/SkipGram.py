import torch


class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, sentences=None, window_size=2, word_to_idx={}, vocab_size=None, num_neg_samples=5):
        assert sentences is not None
        self.vocab_size = vocab_size
        self.num_neg_samples = num_neg_samples
        data = []
        for sentence in sentences:
            for i in range(window_size, len(sentence) - window_size):
                target = sentence[i]
                context = sentence[i - window_size:i] + sentence[i + 1:i + window_size + 1]
                if target in word_to_idx:
                    target_index = word_to_idx[target]
                    for word in context:
                        if word in word_to_idx:
                            data.append((target_index, word_to_idx[word]))
        self.data = data  # 目标词，语境中的词

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target, context = self.data[idx]
        target = torch.tensor(target, dtype=torch.long)
        context = torch.tensor(context, dtype=torch.long)
        neg_samples = torch.randint(0, self.vocab_size, (self.num_neg_samples,), dtype=torch.long)  # TODO
        return target, context, neg_samples


class SkipGramModel(torch.nn.Module):
    """
    SkipGram目标是最大化目标词和其上下文词的相似度
    """

    def __init__(self, vocab_size=None, embedding_dim=100, initial_embeddings=None):
        super(SkipGramModel, self).__init__()
        assert vocab_size is not None
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # 初始化嵌入层：上下文单词嵌入向量、目标词嵌入向量
        if initial_embeddings is not None and len(initial_embeddings) > 0:
            self.context_embeddings = torch.nn.Embedding.from_pretrained(
                torch.tensor(initial_embeddings, dtype=torch.float32), freeze=False)
            self.target_embeddings = torch.nn.Embedding.from_pretrained(
                torch.tensor(initial_embeddings, dtype=torch.float32), freeze=False)
        else:
            self.context_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
            self.target_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
            # 初始化
            for param in self.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

    def forward(self, positive_context, positive_target, negative_context=None, negative_target=None):
        # 获取目标词的嵌入向量
        positive_context_embeds = self.context_embeddings(positive_context)  # Shape: (batch_size, embedding_dim)

        # 获取语境词的嵌入向量
        positive_target_embeds = self.target_embeddings(positive_target)  # Shape: (batch_size, embedding_dim)

        # 计算正样本的点积
        positive_dot_products = torch.sum(positive_target_embeds * positive_context_embeds,
                                          dim=1)  # Shape: (batch_size)

        # 负对数似然损失（Negative Log-Likelihood Loss）：正样本的目标词和上下文词的点积较大
        positive_loss = -torch.mean(torch.log(torch.sigmoid(positive_dot_products)))

        if negative_context is not None and negative_target is not None:
            # Shape: (batch_size * negative, embedding_dim)
            negative_context_embeds = self.context_embeddings(negative_context)
            # Shape: (batch_size * negative, embedding_dim)
            negative_target_embeds = self.target_embeddings(negative_target)

            # 计算负样本的点积：Shape: (batch_size * negative)
            negative_dot_products = torch.sum(negative_target_embeds * negative_context_embeds, dim=1)
            negative_loss = -torch.mean(torch.log(torch.sigmoid(-negative_dot_products)))

        loss = positive_loss + negative_loss
        return loss
