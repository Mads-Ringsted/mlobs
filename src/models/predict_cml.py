import argparse
import sys

import click
import torch
from model import MyAwesomeModel
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



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
        preds, target = [], []
        for images, labels in testloader:
            log_ps = model(images)
            probs = model(images)
            preds.append(probs.argmax(dim=-1))
            target.append(labels.detach())
        model.train()
    
    target = torch.cat(target, dim=0)
    preds = torch.cat(preds, dim=0)
    report = classification_report(target, preds)
    with open("reports/classification_report.txt", 'w') as outfile:
        outfile.write(report)
    confmat = confusion_matrix(target, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix = confmat, display_labels=range(10))
    disp.plot()
    plt.savefig('reports/figures/confusion_matrix.png')



if __name__ == "__main__":
    evaluate()

