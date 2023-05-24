import torch
import torch.nn as nn
import torch.optim as op
from torch_geometric.nn import global_add_pool
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
from torch_scatter import scatter_add


# from torch_geometric.transforms import AddRandomWalkPE


class BasicMPNN(nn.Module):
    def __init__(self, feat_in, edge_feat_in, num_hidden, num_layers):
        super().__init__()
        self.embed = nn.Linear(feat_in, num_hidden)
        self.edge_embed = nn.Linear(edge_feat_in, num_hidden)
        self.layers = nn.ModuleList([BasicMPNNLayer(num_hidden) for _ in range(num_layers)])
        self.predict = nn.Linear(num_hidden, 1)

    def forward(self, graph):
        #print(graph.x, graph.pos)
        h, edge_index, edge_attr,  batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch

        # h_nodes, h_edges, h_triangles
        h = h.float()
        h = self.embed(h)
        edge_attr = edge_attr.unsqueeze(1).float()
        edge_attr = self.edge_embed(edge_attr)

        for layer in self.layers:
            # h_nodes, h_edges, h_triangles = layer(..., edge_nodes_nodes, edge_nodes_edges, ...)
            h = h + nn.functional.relu(layer(h, edge_index, edge_attr))

        h_agg = global_add_pool(h, batch)
        final_prediction = self.predict(h_agg)

        return final_prediction.squeeze(1)


class BasicMPNNLayer(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.message_mlp = nn.Linear(3 * num_hidden, num_hidden)
        self.update_mlp = nn.Linear(2 * num_hidden, num_hidden)

    def forward(self, h, edge_index, edge_attr):
        send, rec = edge_index
        h_send, h_rec = h[send], h[rec]
        messages = self.message_mlp(torch.cat((h_send, h_rec, edge_attr), dim=1))
        messages_agg = scatter_add(messages, rec, dim=0)
        out = self.update_mlp(torch.cat((h, messages_agg), dim=1))

        return out

# transform = PEAddWR(....)
# transform = AddRandomWalkPE(walk_length=4)
data = ZINC('datasets/ZINC_basic') #QM9('datasets/QM9', pre_transform=transform)

train_loader = DataLoader(data[:10], batch_size=32)
val_loader = DataLoader(data[10:12], batch_size=32)
test_loader = DataLoader(data[12:14], batch_size=32)

model = BasicMPNN(11, 3, 32, 4)
optimizer = op.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss(reduce='sum')

for _ in range(10):
    # train
    train_loss = 0
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        print(batch.y.size())
        label = batch.y  # alpha

        out = model(batch)

        loss = criterion(out, label)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(train_loss / len(train_loader.dataset))
# test

model.eval()
test_loss = 0
for batch in test_loader:
    out = model(batch)
    label = batch.y[:, 1]
    loss = criterion(out, label)

    test_loss += loss.item()

print(test_loss / len(test_loader.dataset))