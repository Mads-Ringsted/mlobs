import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel

from torch import nn, optim


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-4, help='learning rate to use for training')
@click.option("--epochs", default=20, help='number of epochs to train for')
@click.option("--model_checkpoint", default='trained_model.pt', help = 'path to save model checkpoint')
def train(lr, epochs, model_checkpoint):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainset, _ = mnist()
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
    torch.save(model, model_checkpoint)

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    with torch.no_grad():
        model.eval()
        test_loss = 0
        accuracy = 0
        for images, labels in testloader:
            log_ps = model(images)

            top_p, top_class = log_ps.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        accuracy = accuracy/len(testloader)
        print(f"Acuracy: {accuracy.item()*100}%")
        model.train()



cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    