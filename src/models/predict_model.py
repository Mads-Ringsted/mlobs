import argparse
import sys
import torch
import click
from model import MyAwesomeModel
from torch import nn, optim


@click.group()
def cli():
    pass

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    test_set = torch.load('data/processed/testset.pt')
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


cli.add_command(evaluate)

if __name__ == "__main__":
    cli()


