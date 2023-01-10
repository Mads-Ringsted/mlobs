import argparse
import sys
import matplotlib.pyplot as plt
import torch
import click

from src.models.model import MyAwesomeModel

from torch import nn, optim


@click.command()
@click.option("--lr", default=1e-4, help='learning rate to use for training')
@click.option("--epochs", default=20, help='number of epochs to train for')
@click.option("--model_checkpoint", default='trained_model.pt', help = 'path to save model checkpoint')
def train(lr, epochs, model_checkpoint=None):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainset = torch.load('data/processed/trainset.pt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    training_loss = []

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            training_loss.append(running_loss/len(trainloader))
            print(f"Training loss: {running_loss/len(trainloader)}")
    
    plt.plot(range(epochs), training_loss)
    plt.savefig('src/visualization/training_curve.png')
    if model_checkpoint is not None:
        torch.save(model, model_checkpoint)
    

if __name__ == "__main__":
    train()