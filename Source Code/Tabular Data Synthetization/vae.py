import torch

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        classes = [9, 16, 7, 15, 6, 5, 2, 42]

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(22, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 8),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
        )
        self.emb = torch.nn.ModuleList([
            torch.nn.Embedding(x, 2)
            for x in classes
        ])
        self.numerical_fc = torch.nn.Linear(64, 4)
        self.linear_fc = torch.nn.Linear(64, 2)
        self.classification = torch.nn.ModuleList([
            torch.nn.Linear(64, x)
            for x in classes
        ])

    def sample(self, encoder_output):
        means = encoder_output[:, :4]
        stds = encoder_output[:, 4:]
        return (
            means,
            stds,
            means + torch.exp(.5 * stds) * torch.randn(
                size = (encoder_output.size(0), 4),
                device = 'cpu'
            )
        )

    def forward(self, numerical, linear, categorical, sample = False):
        input = torch.cat([
            numerical,
            linear,
            *[
                self.emb[i](categorical[:, i])
                for i in range(8)
            ]
        ], axis = -1)
        if sample:
            means, stds, encodings = self.sample(
                self.encoder(input)
            )
        else:
            means, stds = None, None
            encodings = self.encoder(input)[:, :4]
        output = self.decoder(encodings)

        numerical_output = self.numerical_fc(output)
        linear_output = self.linear_fc(output)
        categorical_output = [
            self.classification[i](output)
            for i in range(8)
        ]

        return numerical_output, linear_output, categorical_output, means, stds

    def generate(self, encodings):
        output = self.decoder(encodings)

        numerical_output = self.numerical_fc(output)
        linear_output = self.linear_fc(output)
        categorical_output = [
            self.classification[i](output)
            for i in range(8)
        ]

        return numerical_output, linear_output, categorical_output



