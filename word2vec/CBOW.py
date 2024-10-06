import torch


class CBOWDataset(torch.utils.data.Dataset):
    def __init__(self, sentences=None, window_size=2, word_to_idx={}, vocab_size=None, num_neg_samples=5):
        assert sentences is not None
        self.vocab_size = vocab_size
        self.num_neg_samples = num_neg_samples
        data = []
        for sentence in sentences:
            for i in range(window_size, len(sentence) - window_size):
                target = sentence[i]
                context = ([sentence[j] for j in range(i - window_size, i) if sentence[j] in word_to_idx] +
                           [sentence[j] for j in range(i + 1, i + window_size + 1) if sentence[j] in word_to_idx])
                if target in word_to_idx and context:
                    data.append(([word_to_idx[word] for word in context], word_to_idx[target]))
        self.data = data  # 上下文语境，目标词

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context = torch.tensor(context, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        neg_samples = torch.randint(0, self.vocab_size, (self.num_neg_samples,), dtype=torch.long)  # TODO
        return context, target, neg_samples


class CBOWModel(torch.nn.Module):
    """
    CBOW目标是根据上下文词预测中心词
    """

    def __init__(self, vocab_size=None, embedding_dim=100, initial_embeddings=None):
        super(CBOWModel, self).__init__()
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
        # 获取上下文词的嵌入向量并求平均：# Shape: (batch_size, window_size * 2, embedding_dim)
        positive_context_embeds = self.context_embeddings(positive_context)
        # Shape: (batch_size, embedding_dim)
        positive_context_embeds = positive_context_embeds.mean(dim=1)

        # 获取中心词的嵌入向量：# Shape: (batch_size, embedding_dim)
        positive_target_embeds = self.target_embeddings(positive_target)

        # 计算正样本的点积：Shape: (batch_size)
        positive_dot_products = torch.sum(positive_target_embeds * positive_context_embeds, dim=1)

        # 负对数似然损失（Negative Log-Likelihood Loss）：正样本的目标词和上下文词的点积较大
        positive_loss = -torch.mean(torch.log(torch.sigmoid(positive_dot_products)))

        if negative_context is not None and negative_target is not None:
            # Shape: (batch_size * negative, window_size * 2,  embedding_dim)
            negative_context_embeds = self.context_embeddings(negative_context)
            # Shape: (batch_size * negative, embedding_dim)
            negative_context_embeds = negative_context_embeds.mean(dim=1)

            # Shape: (batch_size * negative, embedding_dim)
            negative_target_embeds = self.target_embeddings(negative_target)

            # 计算负样本的点积：Shape: (batch_size * negative)
            negative_dot_products = torch.sum(negative_target_embeds * negative_context_embeds, dim=1)
            negative_loss = -torch.mean(torch.log(torch.sigmoid(-negative_dot_products)))

        loss = positive_loss + negative_loss
        return loss
