import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from scipy import linalg
import time

# ---------------------- Config ----------------------
PCA_COMPONENTS = 100
LOGREG_LR = 0.02
LOGREG_REG = 1e-5
LOGREG_BATCH = 256
LOGREG_EPOCHS = 400
LOGREG_PATIENCE = 30

TRAIN_CSV = r"D:\IITM SUB\sem 3\ML_LAB\assing\MNIST_train.csv"
VAL_CSV = r"D:\IITM SUB\sem 3\ML_LAB\assing\MNIST_validation.csv"

rng = np.random.default_rng(50)

# ---------------------- Load & Preprocess ----------------------
def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["even"])
    y = df["label"].values.astype(int)
    X = df.drop(columns=["label"]).values.astype(np.float64)
    X /= 255.0
    return X, y

X_train, y_train = load_data(TRAIN_CSV)
X_val, y_val = load_data(VAL_CSV)

# ---------------------- PCA (SVD-Based) ----------------------
class PCA_Scratch:
    def __init__(self, n_components=100):
        self.n_components = n_components

    def fit(self, X):
        n = X.shape[0]
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, S, Vt = linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        self.explained_variance_ = (S**2) / (n - 1)
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

pca = PCA_Scratch(PCA_COMPONENTS)
X_train_pca = pca.fit(X_train).transform(X_train)
X_val_pca = pca.transform(X_val)

# ---------------------- Softmax Logistic Regression ----------------------
class SoftmaxLogisticRegression:
    def __init__(self, n_features, n_classes, lr, reg, batch_size, epochs, patience):
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.W = 0.01 * rng.standard_normal((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

    def softmax(self, Z):
        Z -= Z.max(axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / expZ.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.softmax(X @ self.W + self.b), axis=1)

    def fit(self, X, y, Xv, yv):
        n = len(X)
        Y_onehot = np.zeros((n, self.W.shape[1]))
        Y_onehot[np.arange(n), y] = 1

        best = -1
        no_imp = 0

        for ep in range(self.epochs):
            idx = rng.permutation(n)
            Xs, Ys = X[idx], Y_onehot[idx]

            for i in range(0, n, self.batch_size):
                xb = Xs[i:i+self.batch_size]
                yb = Ys[i:i+self.batch_size]

                logits = xb @ self.W + self.b
                probs = self.softmax(logits)

                grad = (probs - yb) / len(xb)
                gW = xb.T @ grad + self.reg * self.W
                gb = grad.sum(axis=0, keepdims=True)

                lr_t = self.lr / (1 + 0.005 * ep)
                self.W -= lr_t * gW
                self.b -= lr_t * gb

            pred = self.predict(Xv)
            f1 = f1_score(yv, pred, average="weighted")

            if f1 > best:
                best = f1
                no_imp = 0
            else:
                no_imp += 1

            if no_imp >= self.patience:
                break

# ---------------------- Train ----------------------
model = SoftmaxLogisticRegression(
    n_features=X_train_pca.shape[1],
    n_classes=len(np.unique(y_train)),
    lr=LOGREG_LR,
    reg=LOGREG_REG,
    batch_size=LOGREG_BATCH,
    epochs=LOGREG_EPOCHS,
    patience=LOGREG_PATIENCE
)

t0 = time.time()
model.fit(X_train_pca, y_train, X_val_pca, y_val)
t1 = time.time()

pred = model.predict(X_val_pca)

print("Weighted F1:", f1_score(y_val, pred, average="weighted"))
print("Accuracy:", accuracy_score(y_val, pred))
print("Training Time:", round(t1 - t0, 2), "s")
