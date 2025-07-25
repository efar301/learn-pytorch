from tqdm import tqdm
from neural_network import Neural_Network
from data import test_dl, train_dl

import torch.nn as nn
import torch.optim as optim
import argparse

def main(args):
    model = Neural_Network()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs

    for epoch in tqdm(range(num_epochs)):
        for batch in train_dl:
            inputs, targets = batch

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        print(f' Epoch {epoch + 1} / 5 complete. Last batch loss: {loss.item():.4f}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs to train for, default is 5"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate for training, default is 0.001"
    )
    args = parser.parse_args()
    main(args)