import torch.nn as nn


class Toxic(nn.Module):
    def __init__(self, emb_size, num_features, max_len,  num_classes, window_sizes=[1, 2, 3, 4, 5]):
        super(Toxic, self).__init__()

        self.embedding = NavecEmbedding(
            Navec.load('./emb/navec_hudlit_v1_12B_500K_300d_100q.tar')
        )

        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=emb_size,
                    out_channels=num_features,
                    kernel_size=k_size
                ),
                nn.BatchNorm1d(num_features=num_features),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=max_len - k_size + 1)
            ) for k_size in window_sizes
        ])

        self.fc = nn.Linear(
            in_features=num_features * len(window_sizes),
            out_features=num_classes
        )

    def forward(self, x):
        embedded_x = self.embedding(x)
        embedded_x = embedded_x.permute(0, 2, 1)

        out = [conv(embedded_x) for conv in self.conv]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))

        out = F.dropout(
            input=out,
            p=.2
        )

        out = self.fc(out)

        return out
