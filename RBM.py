import torch
from tqdm import tqdm


class RestrictedBoltzmannMachine(torch.nn.Module):
    def __init__(self, hidden_units, visible_units):
        super(RestrictedBoltzmannMachine, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_units = hidden_units
        self.visible_units = visible_units

        self.W = torch.rand(size=(self.hidden_units, self.visible_units), device=self.device)
        self.c, self.b = torch.zeros(size=(self.hidden_units, 1), device=self.device), torch.zeros(
            size=(self.visible_units, 1), device=self.device)

    def hidden_to_visible(self, hidden):
        if hidden.shape[1] == 1:  # Single datapoint
            return torch.matmul(torch.transpose(self.W, 0, 1), hidden) + self.b
        _ones = torch.ones(size=(1, hidden.shape[0]), device=self.device)
        return torch.matmul(hidden, self.W) + torch.transpose(torch.matmul(self.b, _ones), 0, 1)

    def visible_to_hidden(self, visible):
        if visible.shape[1] == 1:  # Single datapoint
            return torch.matmul(self.W, visible) + self.c
        _ones = torch.ones(size=(1, visible.shape[0]), device=self.device)
        return torch.matmul(visible, torch.transpose(self.W, 0, 1)) + torch.transpose(torch.matmul(self.c, _ones), 0, 1)

    @staticmethod
    def sample(vector):
        """returns bit-string sample (as torch tensor) given vector of probabilities."""
        return torch.bernoulli(vector)

    def contrastive_divergence(self, dataset, k: int = 10):

        delta_W = torch.zeros_like(self.W, device=self.device)
        delta_b = torch.zeros_like(self.b, device=self.device)
        delta_c = torch.zeros_like(self.c, device=self.device)

        _v_0 = dataset
        _v_t = _v_0
        _v_k = None
        for k_step in range(0, k):
            # Getting P(h_t|v_t)
            P_h_t = torch.sigmoid(self.visible_to_hidden(visible=_v_t))
            # Sampling h_t ~ P(h_t|v_t)
            _h_t = self.sample(P_h_t)
            # Getting P(v_t|h_t)
            P_v_t = torch.sigmoid(self.hidden_to_visible(hidden=_h_t))
            # Sampling v_t ~ P(h_t|v_t)
            _v_t = self.sample(P_v_t)
            if k_step == k - 1:
                _v_k = _v_t

        # Updating gradients
        P_h_0 = torch.sigmoid(self.visible_to_hidden(visible=_v_0))
        P_h_k = torch.sigmoid(self.visible_to_hidden(visible=_v_k))
        delta_c += torch.sum(P_h_0, dim=0).reshape((self.hidden_units, 1)) - torch.sum(P_h_k, dim=0).reshape(
            (self.hidden_units, 1))
        delta_b += torch.sum(_v_0, dim=0).reshape((self.visible_units, 1)) - torch.sum(_v_k, dim=0).reshape(
            (self.visible_units, 1))
        for _col in range(self.visible_units):
            v_0_col_mat = torch.matmul(_v_0[:, _col].reshape((dataset.shape[0], 1)),
                                       torch.ones(size=(1, self.hidden_units), device=self.device))
            v_k_col_mat = torch.matmul(_v_k[:, _col].reshape((dataset.shape[0], 1)),
                                       torch.ones(size=(1, self.hidden_units), device=self.device))
            delta_W[:, _col] += torch.sum((P_h_0 * v_0_col_mat), dim=0) - torch.sum((P_h_k * v_k_col_mat), dim=0)

        return delta_c, delta_b, delta_W

    def train_network(self, dataset: torch.Tensor,
                      lr: float = 0.001,
                      epochs: int = 100,
                      batch_size: int = 64,
                      k: int = 100):
        assert dataset.shape[
                   1] == self.visible_units, "nr. features in dataset doesn't correspond to nr. visible units."
        assert dataset.device.type == self.device.type, f"dataset: {dataset.device.type} not on same hardware device as RBM network: {self.device.type}."
        print("Using device: ", self.device)
        _N = dataset.shape[0]
        _NR_batches = int(_N / batch_size)
        for epoch in tqdm(range(epochs)):
            for i in range(_NR_batches):
                batch = dataset[i * batch_size:(i + 1) * batch_size]
                delta_c, delta_b, delta_W = self.contrastive_divergence(batch, k)
                self.c += lr * (delta_c / float(batch_size))
                self.b += lr * (delta_b / float(batch_size))
                self.W += lr * (delta_W / float(batch_size))

    def gibbs_sample(self, _v, steps: int = 1):
        """sampling input w. current parameters"""
        # Reshaping to column vector
        _v_0 = _v
        _v_t = _v_0
        for k_step in range(0, steps):
            # Getting P(h_t|v_t)
            P_h_t = torch.sigmoid_(self.visible_to_hidden(visible=_v_t))
            # Sampling h_t ~ P(h_t|v_t)
            _h_t = self.sample(P_h_t)
            # Getting P(v_t|h_t)
            P_v_t = torch.sigmoid_(self.hidden_to_visible(hidden=_h_t))
            # Sampling v_t ~ P(h_t|v_t)
            #_v_t = self.sample(P_v_t)
        return _v_t.flatten()
