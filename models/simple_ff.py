import torch
import torch.nn as nn
import torch.nn.functional as F
import util

def RELU(x):
    return x.relu()

def POOL(x):
    return F.max_pool2d(x, 2)

def POOL1d(x):
    return F.max_pool1d(x, 2)

class DeepNet(nn.Module):
    def __init__(self, D_in, D_out, HS, dropout_rate, batchnorm=False):
        super(DeepNet, self).__init__()
        self.fc1 = nn.Linear(D_in, HS[0])
        self.bn1 = nn.BatchNorm1d(HS[0])

        self.fc2 = nn.Linear(HS[0], HS[1])
        self.bn2 = nn.BatchNorm1d(HS[1])

        self.fc3 = nn.Linear(HS[1], HS[2])
        self.bn3 = nn.BatchNorm1d(HS[2])

        self.fc4 = nn.Linear(HS[2], D_out)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

        self.batchnorm = batchnorm

    def forward(self, x):
        if self.batchnorm:
            x = self.drop1(RELU(self.bn1(self.fc1(x))))
            x = self.drop2(RELU(self.bn2(self.fc2(x))))
            x = RELU(self.bn3(self.fc3(x)))

        else:
            x = self.drop1(RELU(self.fc1(x)))
            x = self.drop2(RELU(self.fc2(x)))
            x = RELU(self.fc3(x))

        pred = self.fc4(x)
        
        return pred
    
class MTLRNN(nn.Module):
    def __init__(self, D_in, num_cat_bins, **kwargs):
        super(MTLRNN, self).__init__()
        # IN MTLR, the prob of first bin is exp(0)
        self.fc = nn.Linear(D_in, num_cat_bins-1)
        self.kwargs = kwargs
        self.D_in = D_in

    def forward(self, x):
        preds = self.fc(x)
        preds = util.cumsum_reverse(preds, dim=1)
        
        return preds

class AFTNN(nn.Module):
    def __init__(self, data_dir, D_in, model_dist, big_gamma_nn, num_cat_bins, hiddensize, dropout_rate, **kwargs):
        super(AFTNN, self).__init__()

        self.kwargs = kwargs
        self.data_dir = data_dir
        self.D_in = D_in
        self.beta = nn.Parameter(torch.randn(D_in))
        self.pre_k = nn.Parameter(torch.tensor([1.0]))

    def smart_init_2(self):
        self.beta.data = torch.load('models/beta.pt')
        self.pre_k.data = torch.load('models/shape.pt')

    def smart_init(self, train_loader):
        """
        fit a lifelines AFT model
        """
        from lifelines import WeibullAFTFitter
        import pandas as pd

        dataset = train_loader.dataset

        x = dataset.src
        tgt = dataset.tgt

        t = tgt[:, 0]
        c = tgt[:, 1]

        print("t", t.size())
        print("c", c.size())
        print("x", x.size())

        tcx = torch.cat((t.unsqueeze(-1), c.unsqueeze(-1), x), dim=-1)
        print("tcx", tcx.size())
        tcx_numpy = tcx.numpy()

        col_names = ["T", "C"]
        for i in range(x.size()[-1]):
            col_names.append("f" + str(i))

        df = pd.DataFrame(data=tcx_numpy, columns=col_names)
        print("df shape", df.shape)

        print("initializing aft model with lifelines aft fit")

        aft = WeibullAFTFitter()
        aft.fit(df, 'T', 'C')
        aft.print_summary()

        lam = aft.params_.lambda_
        rho = aft.params_.rho_

        lam_list = [float(lam.loc['f{}'.format(i)]) for i in range(x.size()[-1])]

        lam = torch.tensor(lam_list)
        rho = torch.tensor([rho.to_numpy()])

        self.beta.data = lam
        self.pre_k.data = torch.tensor([0.3])
        print("Initialized AFT model with lifelines AFT fit")

    def forward(self, x):
        batch_sz = x.size()[0]
        expbetax = (self.beta*x).sum(-1).exp().view(-1, 1)
        pre_k = self.pre_k.view(-1, 1).repeat(batch_sz, 1)
        
        return torch.cat([expbetax, pre_k], dim=1)

class LogNormalhelper(nn.Module):
    def __init__(self, D_in, D_out, hidden_sizes, dropout_rate, batchnorm=False):
        super(LogNormalhelper, self).__init__()
        self.mu_model = DeepNet(D_in, D_out, hidden_sizes, dropout_rate, batchnorm)
        #self.sigma_model = DeepNet(D_in, D_out, hidden_sizes, dropout_rate, batchnorm)
        
    def forward(self, src):
        mu = self.mu_model(src).view(-1, 1)
        #pre_log_sigma = self.sigma_model(src).view(-1, 1)
        #pred = torch.cat([mu, pre_log_sigma], dim = 1)
        pred = mu

        return pred

class LogNormalMNISTHelper(nn.Module):
    def __init__(self, D_in, D_out, hidden_sizes, dropout_rate):
        super(LogNormalMNISTHelper, self).__init__()
        self.mu_model = DeepNet(D_in, D_out, hidden_sizes, dropout_rate)
        self.sigma_model = DeepNet(D_in, D_out, hidden_sizes, dropout_rate)

    def forward(self, src1, src2):
        mu = self.mu_model(src1).view(-1, 1)
        pre_log_sigma = self.sigma_model(src2).view(-1, 1)
        pred = torch.cat([mu, pre_log_sigma], dim=1)

        return pred

class GammaNN(nn.Module):
    def __init__(self, data_dir, D_in, model_dist, big_gamma_nn, num_cat_bins, hiddensize, dropout_rate, **kwargs):
        super(GammaNN, self).__init__()
        self.kwargs = kwargs
        self.data_dir = data_dir
        self.D_in = D_in
        
        assert model_dist in ['cat', 'lognormal']
        
        self.model_dist = model_dist

        # recent experiments
        #hidden_sizes = [1024, 1024, 1024]

        # re-create original paper results for gamma data with cat model
        hidden_sizes = [128, 64, 64]

        if model_dist == 'cat':
            self.model = DeepNet(D_in, num_cat_bins, hidden_sizes, dropout_rate)

        elif model_dist == 'lognormal':
            self.model = LogNormalhelper(D_in, 1, hidden_sizes, dropout_rate, batchnorm=True)

        else:
            assert False

    def init_weights(self, m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, src):
        pred = self.model(src)

        return pred

    def args_dict(self):
        """
        Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `ModelName(**model_args)`.
        """
        model_args = {'D_in': self.D_in}

        return model_args

class MIMICNN(nn.Module):
    def __init__(self, data_dir, D_in, model_dist, num_cat_bins, dropout_rate, **kwargs):
        super(MIMICNN, self).__init__()
        self.kwargs = kwargs
        self.data_dir = data_dir
        self.D_in = D_in

        # assert model_dist in ['cat', 'mtlr'], "need to update this class for non-cat"

        hidden_sizes = [2048, 1024, 1024]
        # self.model = DeepNet(self.D_in, D_out, ff_hidden_sizes, dropout_rate)

        if model_dist == 'cat':
            self.model = DeepNet(D_in, num_cat_bins, hidden_sizes, dropout_rate)

        elif model_dist == 'mtlr':
            self.model = nn.Sequential(DeepNet(D_in, num_cat_bins, hidden_sizes, dropout_rate), MTLRNN(num_cat_bins, num_cat_bins))

        elif model_dist == 'lognormal':
            self.model = LogNormalhelper(D_in, 1, hidden_sizes, dropout_rate, batchnorm=True)

        else:
            assert False

    def forward(self, src):
        pred = self.model(src)

        return pred

    def args_dict(self):
        """
        Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `ModelName(**model_args)`.
        """
        model_args = {'D_in': self.D_in}

        return model_args

class SimpleNN(nn.Module):
    def __init__(self, data_dir, D_in, device, dropout_rate, model_dist, num_cat_bins, use_max_pool, **kwargs):
        super(SimpleNN, self).__init__()

        self.data_dir = data_dir
        self.D_in = D_in
        self.num_demographics = 2
        self.vocab_size = D_in - self.num_demographics

        EMBED_SIZE = 128
        hidden_sizes = [512, 1024, 1024]
        self.embedding_dim = EMBED_SIZE
        if use_max_pool:
            input_dim = self.num_demographics + self.embedding_dim*2
        else:
            input_dim = self.num_demographics + self.embedding_dim

        if model_dist in ['cat','mtlr']:
            self.model = DeepNet(D_in=input_dim, D_out=num_cat_bins, HS=hidden_sizes, dropout_rate=dropout_rate, batchnorm=True)
        elif model_dist == 'lognormal':
            self.model = LogNormalhelper(D_in*4 + self.num_demographics, 1, hidden_sizes, dropout_rate, batchnorm=True)
            
        
        else:
            assert False, 'only cat/mtlr/lognormal allowed as model dist'
        # self.model.apply(self.init_weights)
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, HIDDEN_ONE),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(HIDDEN_ONE, HIDDEN_TWO),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(HIDDEN_TWO, HIDDEN_THREE),
        #     nn.ReLU(),
        #     nn.Linear(HIDDEN_THREE, num_classes),
        # )
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embed.to(device)
        self.use_max_pool = use_max_pool
        self.kwargs = kwargs

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, src):
        src_dem = src[:, :self.num_demographics]  # (batch x 2)
        src_codes = src[:, self.num_demographics:].long()  # (batch x 6798)
        
        # list of length batch sz, each entry is (some amount of labs x 128 codes)
        # embedded_codes = [self.embed(torch.nonzero(c)[:,0]) for c in src_codes]

        # list of length batch size, each entry is (1, 128)
        # embedded_codes = [e.mean(dim=0).unsqueeze(0) for e in embedded_codes]

        # but instead do this in one loop, list is length batch size, each entry (1,128)
        embedded_codes = [self.embed(torch.nonzero(c)[:, 0]).mean(dim=0).unsqueeze(0) for c in src_codes]
        
        # (batch sz, 128)
        embedded_codes = torch.cat(embedded_codes)

        src = torch.cat((src_dem, embedded_codes), dim=1)
        if self.use_max_pool:
            embedded_codes_max = [self.embed(torch.nonzero(c)[:, 0]).max(dim=0)[0].unsqueeze(0) for c in src_codes]
            embedded_codes_max = torch.cat(embedded_codes_max)
            src = torch.cat((src, embedded_codes_max), dim=1)
        pred = self.model(src)
        
        return pred

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `ModelName(**model_args)`.
        """
        model_args = {
            'D_in': self.D_in,
        }

        return model_args