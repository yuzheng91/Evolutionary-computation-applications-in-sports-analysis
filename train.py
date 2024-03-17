import os
import torch
import torch.nn as nn
import numpy as np
from model import Model
from load import get_feature
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import LeaveOneOut


def setseed(manualSeed):
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    torch.backends.cudnn.deterministic = True


def CV(x, wr, num_feat):
    if not os.path.exists("result"):
        os.makedirs("result")

    loocv = LeaveOneOut()

    avg_loss = 0.0

    for train_index, test_index in loocv.split(x):
        x_ax_train = []
        y_ax_train = []
        x_ax_test = []
        y_ax_test = []

        x_train = x[train_index]
        y_train = wr[train_index]
        x_test = x[test_index]
        y_test = wr[test_index]

        print(x_test)

        model = Model(num_feat)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(epochs):
            prediction = model(x_train)
            loss = loss_func(prediction, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            x_ax_train.append(epoch)
            y_ax_train.append(loss.item())

            model.eval()
            with torch.no_grad():
                prediction = model(x_test)
                loss = loss_func(prediction, y_test)
                x_ax_test.append(epoch)
                y_ax_test.append(loss.item())

                avg_loss += loss.item()

        # print(epoch, loss.item(), prediction)

        plt.plot(x_ax_train, y_ax_train, label="Train")
        plt.plot(x_ax_test, y_ax_test, label="Test")
        plt.legend()

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # plt.show()
        plt.savefig(f"result/{test_index[0]}.png")

        plt.clf()

    print(avg_loss / num_team)


def Train(x, wr, num_feat):
    model = Model(num_feat)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x_ax_train = []
    y_ax_train = []

    model.train()
    for epoch in range(epochs):
        prediction = model(x)
        loss = loss_func(prediction, wr)

        x_ax_train.append(epoch)
        y_ax_train.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "model.pt")

    plt.plot(x_ax_train, y_ax_train, label="Train")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.show()
    plt.savefig("Training.png")


if __name__ == "__main__":
    manualSeed = 2
    setseed(manualSeed)

    device = torch.device("cuda")

    x, wr, num_feat = get_feature()
    # x = normalize(x)
    # x = StandardScaler().fit_transform(x.T).T
    x = torch.Tensor(x)
    x = x.to(device)

    num_team = len(wr)
    print(f"Teams: {num_team}")
    print(f"Feature: {num_feat}")

    wr = torch.Tensor(wr)
    wr = wr.to(device)

    epochs = 200
    loss_func = nn.MSELoss()

    CV(x, wr, num_feat)
    # Train(x, wr, num_feat)
