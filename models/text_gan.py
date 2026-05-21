import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, z_dim=64, hidden_dim=128, seq_len=20):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        
        self.rnn = nn.GRU(z_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, tau=1.0):
        # z: (batch_size, z_dim)
        batch_size = z.size(0)
        
        # Повторяем z для каждого шага: (batch_size, seq_len, z_dim)
        z_seq = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Пропускаем через RNN
        # h0 инициализируется нулями по умолчанию
        out, _ = self.rnn(z_seq) # out: (batch_size, seq_len, hidden_dim)
        
        # Получаем логиты для каждого токена
        logits = self.fc(out) # (batch_size, seq_len, vocab_size)
        
        # Gumbel-Softmax позволяет прокидывать градиенты через "дискретный" выбор
        prob = F.gumbel_softmax(logits, tau=tau, hard=False)
        return prob

class TextDiscriminator(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, num_filters=64, filter_sizes=[2, 3, 4, 5]):
        super().__init__()
        # Embedding matrix: (vocab_size, emb_dim)
        self.emb = nn.Linear(vocab_size, emb_dim, bias=False) 
        # Мы используем Linear вместо nn.Embedding, чтобы можно было умножать 
        # continuous векторы от Gumbel-Softmax на матрицу эмбеддингов.
        
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)
        
    def forward(self, x, is_discrete=False):
        # x может быть либо (batch, seq_len) с индексами токенов (real data)
        # либо (batch, seq_len, vocab_size) с вероятностями от генератора (fake data)
        if is_discrete:
            # Превращаем индексы в one-hot
            batch_size, seq_len = x.size()
            x_one_hot = F.one_hot(x, num_classes=self.emb.in_features).float()
            e = self.emb(x_one_hot) # (batch, seq, emb_dim)
        else:
            e = self.emb(x) # (batch, seq, emb_dim)
            
        # Conv1d ожидает (batch, channels, seq)
        e = e.transpose(1, 2)
        
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(e)) # (batch, num_filters, seq_len - fs + 1)
            p = F.max_pool1d(c, c.size(2)).squeeze(2) # (batch, num_filters)
            pooled.append(p)
            
        cat = torch.cat(pooled, dim=1)
        return self.fc(cat)

def train_text_gan(gen, disc, dataloader, vocab_size, n_epochs=50, lr=1e-3, device='cpu'):
    gen.to(device)
    disc.to(device)
    
    opt_G = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    
    history = {'d_loss': [], 'g_loss': []}
    tau = 1.0 # Температура для Gumbel-Softmax
    
    gen.train()
    disc.train()
    
    from tqdm.auto import tqdm
    pbar = tqdm(range(n_epochs))
    
    for epoch in pbar:
        d_loss_epoch = 0
        g_loss_epoch = 0
        
        # Понижаем температуру, чтобы ответы генератора становились более дискретными (one-hot)
        tau = max(0.5, tau * 0.95)
        
        for batch in dataloader:
            real_data = batch[0].to(device) # (batch, seq_len)
            batch_size = real_data.size(0)
            
            # 1. Train Discriminator
            opt_D.zero_grad()
            
            # Real
            real_logits = disc(real_data, is_discrete=True)
            loss_D_real = criterion(real_logits, torch.ones_like(real_logits))
            
            # Fake
            z = torch.randn(batch_size, gen.z_dim, device=device)
            fake_probs = gen(z, tau=tau)
            fake_logits = disc(fake_probs.detach(), is_discrete=False)
            loss_D_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
            
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            opt_D.step()
            
            # 2. Train Generator
            # Тренируем генератор обманывать дискриминатор
            opt_G.zero_grad()
            z = torch.randn(batch_size, gen.z_dim, device=device)
            fake_probs = gen(z, tau=tau)
            fake_logits = disc(fake_probs, is_discrete=False)
            
            loss_G = criterion(fake_logits, torch.ones_like(fake_logits))
            loss_G.backward()
            opt_G.step()
            
            d_loss_epoch += loss_D.item()
            g_loss_epoch += loss_G.item()
            
        d_loss_epoch /= len(dataloader)
        g_loss_epoch /= len(dataloader)
        
        history['d_loss'].append(d_loss_epoch)
        history['g_loss'].append(g_loss_epoch)
        
        pbar.set_description(f"D={d_loss_epoch:.4f} G={g_loss_epoch:.4f} tau={tau:.2f}")
        
    return history
