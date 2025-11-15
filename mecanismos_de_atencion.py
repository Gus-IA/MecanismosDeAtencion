import numpy as np
import torch
import matplotlib.pyplot as plt

# ---- Hard Attention ----

X = torch.tensor([[1, 0], [0, 1], [-0.5, -0.5]])
print(X)


def plot_vectors(X, y=None):
    ax = plt.axes()
    plt.grid()
    for x in X:
        ax.arrow(0, 0, x[0], x[1], head_width=0.1, head_length=0.1, color="black")
    if y is not None:
        for _y in y:
            if _y.sum() != 0:
                ax.arrow(
                    0, 0, _y[0], _y[1], head_width=0.1, head_length=0.1, color="red"
                )
    plt.xlim(-1, 1.5)
    plt.ylim(-1, 1.5)
    plt.show()


plot_vectors(X)


# hard attention (a es one hot)
# atendemos al primer vector
a = torch.tensor([1, 0, 0])
print(a)

# todos los vectores en la salida son 0, excepto al que hemos prestado atención

y = a.unsqueeze(1) * X
print(y)

plot_vectors(X, y)

A = torch.eye(3)
print(A)

Y = A @ X
print(Y)


def plot(X, Y):
    fig = plt.figure(figsize=(15, 4))
    for i, (x, y) in enumerate(zip(X, Y)):
        ax = plt.subplot(1, 3, i + 1)
        for x in X:
            ax.arrow(0, 0, x[0], x[1], head_width=0.1, color="black", head_length=0.1)
        ax.arrow(0, 0, y[0], y[1], head_width=0.1, head_length=0.1, color="red")
        ax.set_xlim(-1, 1.5)
        ax.set_ylim(-1, 1.5)
        ax.grid(True)
        ax.set_title(f"Atiende a {X[i].tolist()}")
    plt.tight_layout()
    plt.show()


plot(X, Y)


# ---- Soft Attention ----

# soft attention (cada fila suma 1)

A = torch.ones((3, 3)) * 0.1
A.fill_diagonal_(0.8)
print(A)


Y = A @ X
print(Y)


def plot(X, Y):
    fig = plt.figure(figsize=(15, 4))
    for i, (x, y) in enumerate(zip(X, Y)):
        ax = plt.subplot(1, 3, i + 1)
        for x in X:
            ax.arrow(0, 0, x[0], x[1], color="black", head_width=0.1, head_length=0.1)
        Z = A[i].unsqueeze(1) * X
        for z in Z:
            ax.arrow(0, 0, z[0], z[1], head_width=0.1, head_length=0.1, color="green")
        ax.arrow(0, 0, y[0], y[1], head_width=0.1, head_length=0.1, color="red")
        ax.set_xlim(-1, 1.5)
        ax.set_ylim(-1, 1.5)
        ax.grid(True)
        ax.set_title(f"Atiende a {X[i].tolist()}")
    plt.tight_layout()
    plt.show()


plot(X, Y)

A = torch.ones((3, 3)) * (1.0 / 3.0)
print(A)

Y = A @ X
print(Y)

plot(X, Y)

# ---- Self Attention ----


A = torch.softmax(X @ X.T, 1)
print(A)

Y = A @ X
print(Y)

plot(X, Y)
