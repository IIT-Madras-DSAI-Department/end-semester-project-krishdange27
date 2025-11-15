## Imports & config
import time
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

RNG = np.random.default_rng(50)

# Paths - update to your actual file names
TRAIN_CSV = r"D:\IITM SUB\sem 3\ML_LAB\assing\MNIST_train.csv"
VAL_CSV = r"D:\IITM SUB\sem 3\ML_LAB\assing\MNIST_validation.csv"

# Settings
PCA_COMPONENTS = 100


LOGREG_EPOCHS = 300
LOGREG_LR = 0.01
LOGREG_BATCH = 256

RF_N_ESTIMATORS = 25
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 3

GB_N_EST_PER_CLASS = 8
GB_LR = 0.1
GB_MAX_DEPTH = 3

PRINT_VERBOSE = True


## Data load & preprocess
def load_and_preprocess(train_path, val_path, drop_cols=('even',), label_col='label'):
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    for c in drop_cols:
        if c in df_train.columns:
            df_train = df_train.drop(columns=[c])
        if c in df_val.columns:
            df_val = df_val.drop(columns=[c])
    X_train = df_train.drop(columns=[label_col]).values.astype(np.float64)
    y_train = df_train[label_col].values.astype(int)
    X_val = df_val.drop(columns=[label_col]).values.astype(np.float64)
    y_val = df_val[label_col].values.astype(int)
    X_train /= 255.0
    X_val /= 255.0
    return X_train, y_train, X_val, y_val

X_train, y_train, X_val, y_val = load_and_preprocess(TRAIN_CSV, VAL_CSV)

print("Train shape:", X_train.shape, "Val shape:", X_val.shape)


## PCA implementation
class PCA_Scratch:
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, S, Vt = linalg.svd(Xc, full_matrices=False)
        eigenvals = (S**2) / (n_samples - 1)
        self.components_ = Vt[:self.n_components]
        self.explained_variance_ = eigenvals[:self.n_components]
        total_var = eigenvals.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

pca = PCA_Scratch(n_components=PCA_COMPONENTS)
t0 = time.time()
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
t1 = time.time()
print(f"PCA done in {t1-t0:.2f}s. Result shapes:", X_train_pca.shape, X_val_pca.shape)

# print cumulative explained variance for first 10 and total of chosen components
cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"Total explained variance by {PCA_COMPONENTS} components: {cumvar[-1]:.4f}")

# quick plot (optional)
plt.figure(figsize=(5,3))
plt.plot(np.arange(1, len(cumvar)+1), cumvar, marker='o')
plt.xlabel('Component index')
plt.ylabel('Cumulative explained variance')
plt.grid(True)
plt.show()


## Logistic Regression (softmax) from scratch
class SoftmaxLogisticRegression:
    def __init__(self, n_features, n_classes, lr=0.01, reg=1e-4, batch_size=256, epochs=300, verbose=False):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.W = 0.01 * RNG.standard_normal((n_features, n_classes), dtype=np.float64)
        self.b = np.zeros((1, n_classes), dtype=np.float64)

    def _softmax(self, Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def _one_hot(self, y):
        Y = np.zeros((y.shape[0], self.n_classes), dtype=np.float64)
        Y[np.arange(y.shape[0]), y] = 1.0
        return Y

    def predict_proba(self, X):
        Z = X.dot(self.W) + self.b
        return self._softmax(Z)

    def predict(self, X):
        P = self.predict_proba(X)
        return np.argmax(P, axis=1)

    def fit(self, X, y, X_val=None, y_val=None, early_stopping=False, patience=10):
        n = X.shape[0]
        Y = self._one_hot(y)
        history = {'loss': [], 'val_f1': []}
        for epoch in range(self.epochs):
            idx = RNG.permutation(n)
            X_shuf = X[idx]
            Y_shuf = Y[idx]
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                xb = X_shuf[start:end]
                yb = Y_shuf[start:end]
                logits = xb.dot(self.W) + self.b
                probs = self._softmax(logits)
                grad_logits = (probs - yb) / max(1, xb.shape[0])
                gradW = xb.T.dot(grad_logits) + self.reg * self.W
                gradb = np.sum(grad_logits, axis=0, keepdims=True)
                self.W -= self.lr * gradW
                self.b -= self.lr * gradb
            logits_all = X.dot(self.W) + self.b
            probs_all = self._softmax(logits_all)
            eps = 1e-12
            loss = -np.sum(Y * np.log(probs_all + eps)) / n + 0.5 * self.reg * np.sum(self.W**2)
            history['loss'].append(loss)
            if X_val is not None and y_val is not None:
                y_pred_val = self.predict(X_val)
                val_f1 = f1_score(y_val, y_pred_val, average='weighted')
                history['val_f1'].append(val_f1)
                if self.verbose and ((epoch+1) % 50 == 0 or epoch==0):
                    print(f"Epoch {epoch+1}/{self.epochs} loss={loss:.4f} val_f1={val_f1:.4f}")
                if early_stopping and len(history['val_f1']) > patience:
                    if history['val_f1'][-1] <= max(history['val_f1'][-(patience+1):-1]):
                        if self.verbose:
                            print("Early stopping triggered.")
                        break
            else:
                if self.verbose and ((epoch+1) % 50 == 0 or epoch==0):
                    print(f"Epoch {epoch+1}/{self.epochs} loss={loss:.4f}")
        return history

# Train logistic regression on PCA features
n_features = X_train_pca.shape[1]
n_classes = int(max(y_train.max(), y_val.max()) + 1)
logreg = SoftmaxLogisticRegression(n_features=n_features, n_classes=n_classes,
                                   lr=LOGREG_LR, reg=1e-4, batch_size=LOGREG_BATCH, epochs=LOGREG_EPOCHS, verbose=True)
t0 = time.time()
hist_log = logreg.fit(X_train_pca, y_train, X_val=X_val_pca, y_val=y_val, early_stopping=True, patience=10)
t1 = time.time()
train_time = t1 - t0
y_pred_val = logreg.predict(X_val_pca)
f1_log = f1_score(y_val, y_pred_val, average='weighted')
acc_log = accuracy_score(y_val, y_pred_val)


## Decision Tree and Random Forest
class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, n_thresholds=16):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.tree_ = None

    def _gini(self, y):
        if y.size == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1.0 - np.sum(p**2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best = {'feature': None, 'threshold': None, 'score': np.inf, 'left_idx': None, 'right_idx': None}
        if n_samples < self.min_samples_split:
            return best
        for feat in range(n_features):
            col = X[:, feat]
            uniq = np.unique(col)
            if uniq.size <= 1:
                continue
            perc = np.linspace(0, 100, num=min(self.n_thresholds, uniq.size))
            thresholds = np.percentile(col, perc[1:-1])
            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask
                if (left_mask.sum() < self.min_samples_leaf) or (right_mask.sum() < self.min_samples_leaf):
                    continue
                g_left = self._gini(y[left_mask])
                g_right = self._gini(y[right_mask])
                score = (left_mask.sum() * g_left + right_mask.sum() * g_right) / n_samples
                if score < best['score']:
                    best.update({'feature': feat, 'threshold': thr, 'score': score,
                                 'left_idx': left_mask, 'right_idx': right_mask})
        return best

    def _build(self, X, y, depth=0):
        node = {}
        num_samples = X.shape[0]
        num_labels = np.unique(y).size
        if (depth >= self.max_depth) or (num_samples < self.min_samples_split) or (num_labels == 1):
            vals, counts = np.unique(y, return_counts=True)
            node['type'] = 'leaf'
            node['class'] = vals[np.argmax(counts)]
            return node
        split = self._best_split(X, y)
        if split['feature'] is None:
            vals, counts = np.unique(y, return_counts=True)
            node['type'] = 'leaf'
            node['class'] = vals[np.argmax(counts)]
            return node
        node['type'] = 'node'
        node['feature'] = split['feature']
        node['threshold'] = split['threshold']
        left_X = X[split['left_idx']]
        left_y = y[split['left_idx']]
        right_X = X[split['right_idx']]
        right_y = y[split['right_idx']]
        node['left'] = self._build(left_X, left_y, depth+1)
        node['right'] = self._build(right_X, right_y, depth+1)
        return node

    def fit(self, X, y):
        self.tree_ = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node['type'] == 'leaf':
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def predict(self, X):
        if self.tree_ is None:
            raise ValueError("Tree not trained.")
        return np.array([self._predict_one(x, self.tree_) for x in X])

class RandomForestScratch:
    def __init__(self, n_estimators=25, max_depth=12, min_samples_leaf=1, mtry=None, n_thresholds=16):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.mtry = mtry
        self.n_thresholds = n_thresholds
        self.trees = []
        self.trees_features = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.mtry is None:
            self.mtry = max(1, int(math.sqrt(n_features)))
        self.trees = []
        self.trees_features = []
        for t in range(self.n_estimators):
            idx = RNG.integers(0, n_samples, size=n_samples)
            X_sample = X[idx]
            y_sample = y[idx]
            feat_idx = RNG.choice(n_features, size=self.mtry, replace=False)
            self.trees_features.append(feat_idx)
            tree = DecisionTreeClassifierScratch(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, n_thresholds=self.n_thresholds)
            tree.fit(X_sample[:, feat_idx], y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = int(max(y_train.max(), y_val.max()) + 1)
        votes = np.zeros((n_samples, n_classes), dtype=int)
        for tree, feat_idx in zip(self.trees, self.trees_features):
            preds = tree.predict(X[:, feat_idx])
            for i, p in enumerate(preds):
                votes[i, int(p)] += 1
        preds_final = np.argmax(votes, axis=1)
        return preds_final

# Train Random Forest on PCA features
rf = RandomForestScratch(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, min_samples_leaf=RF_MIN_SAMPLES_LEAF, mtry=max(1,int(math.sqrt(n_features))), n_thresholds=16)
t0 = time.time()
rf.fit(X_train_pca, y_train)
t1 = time.time()
rf_train_time = t1 - t0
y_pred_rf = rf.predict(X_val_pca)
f1_rf = f1_score(y_val, y_pred_rf, average='weighted')
acc_rf = accuracy_score(y_val, y_pred_rf)


## Simple regression tree and one-vs-rest gradient booster
class SimpleRegressionTree:
    def __init__(self, max_depth=3, min_samples_leaf=5, n_thresholds=8):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.tree_ = None

    def _mse(self, y):
        if y.size == 0:
            return 0.0
        return np.mean((y - y.mean())**2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best = {'feature': None, 'threshold': None, 'score': np.inf, 'left_idx': None, 'right_idx': None}
        if n_samples < self.min_samples_leaf * 2:
            return best
        for feat in range(n_features):
            col = X[:, feat]
            uniq = np.unique(col)
            if uniq.size <= 1:
                continue
            perc = np.linspace(0, 100, num=min(self.n_thresholds, uniq.size))
            thresholds = np.percentile(col, perc[1:-1])
            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask
                if (left_mask.sum() < self.min_samples_leaf) or (right_mask.sum() < self.min_samples_leaf):
                    continue
                score = (left_mask.sum() * self._mse(y[left_mask]) + right_mask.sum() * self._mse(y[right_mask])) / n_samples
                if score < best['score']:
                    best.update({'feature': feat, 'threshold': thr, 'score': score,
                                 'left_idx': left_mask, 'right_idx': right_mask})
        return best

    def _build(self, X, y, depth=0):
        node = {}
        n_samples = y.size
        if (depth >= self.max_depth) or (n_samples < self.min_samples_leaf * 2):
            node['type'] = 'leaf'
            node['value'] = y.mean() if n_samples > 0 else 0.0
            return node
        split = self._best_split(X, y)
        if split['feature'] is None:
            node['type'] = 'leaf'
            node['value'] = y.mean() if n_samples > 0 else 0.0
            return node
        node['type'] = 'node'
        node['feature'] = split['feature']
        node['threshold'] = split['threshold']
        left_X = X[split['left_idx']]
        left_y = y[split['left_idx']]
        right_X = X[split['right_idx']]
        right_y = y[split['right_idx']]
        node['left'] = self._build(left_X, left_y, depth+1)
        node['right'] = self._build(right_X, right_y, depth+1)
        return node

    def fit(self, X, y):
        self.tree_ = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node['type'] == 'leaf':
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def predict(self, X):
        if self.tree_ is None:
            raise ValueError("Tree not trained.")
        return np.array([self._predict_one(x, self.tree_) for x in X])

class SimpleBinaryGB:
    def __init__(self, n_estimators=8, learning_rate=0.1, max_depth=3, min_samples_leaf=5, n_thresholds=8):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.trees = []
        self.base_score = None

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y, X_val=None, y_val=None, early_stopping=False, patience=3):
        n = X.shape[0]
        pos_rate = np.clip(y.mean(), 1e-6, 1-1e-6)
        F = np.full(n, np.log(pos_rate / (1 - pos_rate)), dtype=np.float64)
        self.base_score = F[0]
        self.trees = []
        best_val_f1 = -1.0
        rounds_without_improve = 0
        for m in range(self.n_estimators):
            p = self._sigmoid(F)
            residual = y - p
            tree = SimpleRegressionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, n_thresholds=self.n_thresholds)
            tree.fit(X, residual)
            update = tree.predict(X)
            F = F + self.learning_rate * update
            self.trees.append(tree)
            if X_val is not None and y_val is not None:
                F_val = np.full(X_val.shape[0], np.log(pos_rate / (1 - pos_rate)), dtype=np.float64)
                for t in self.trees:
                    F_val += self.learning_rate * t.predict(X_val)
                y_pred = (self._sigmoid(F_val) >= 0.5).astype(int)
                val_f1 = f1_score(y_val, y_pred, average='weighted')
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    rounds_without_improve = 0
                else:
                    rounds_without_improve += 1
                if early_stopping and rounds_without_improve >= patience:
                    print(f"Early stopping GB at round {m+1}, best_val_f1={best_val_f1:.4f}")
                    break
        return self

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.base_score, dtype=np.float64)
        for t in self.trees:
            F += self.learning_rate * t.predict(X)
        p = self._sigmoid(F)
        return np.vstack([1-p, p]).T

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)

class OneVsRestGB:
    def __init__(self, n_classes, n_estimators_per_class=8, learning_rate=0.1, max_depth=3):
        self.n_classes = n_classes
        self.n_estimators_per_class = n_estimators_per_class
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = [None] * n_classes

    def fit(self, X, y, X_val=None, y_val=None, early_stopping=False, patience=3):
        for c in range(self.n_classes):
            print(f"Training booster for class {c}")
            y_bin = (y == c).astype(int)
            if X_val is not None and y_val is not None:
                y_val_bin = (y_val == c).astype(int)
            else:
                y_val_bin = None
            model = SimpleBinaryGB(n_estimators=self.n_estimators_per_class, learning_rate=self.learning_rate, max_depth=self.max_depth)
            model.fit(X, y_bin, X_val=X_val, y_val=y_val_bin, early_stopping=early_stopping, patience=patience)
            self.models[c] = model
        return self

    def predict(self, X):
        probs = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)
        for c, model in enumerate(self.models):
            probs[:, c] = model.predict_proba(X)[:, 1]
        return np.argmax(probs, axis=1)

# Train One-vs-Rest GB
ovr_gb = OneVsRestGB(n_classes=n_classes, n_estimators_per_class=GB_N_EST_PER_CLASS, learning_rate=GB_LR, max_depth=GB_MAX_DEPTH)
t0 = time.time()
ovr_gb.fit(X_train_pca, y_train, X_val=X_val_pca, y_val=y_val, early_stopping=True, patience=3)
t1 = time.time()
gb_train_time = t1 - t0
y_pred_gb = ovr_gb.predict(X_val_pca)
f1_gb = f1_score(y_val, y_pred_gb, average='weighted')
acc_gb = accuracy_score(y_val, y_pred_gb)

## Linear SVM (One-vs-Rest) improved training

class LinearSVM_OVR:
    def __init__(self, n_features, n_classes, lr=0.02, C=10.0, epochs=400, batch_size=256, patience=30, verbose=False):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        rng = np.random.default_rng(42)
        self.W = 0.01 * rng.standard_normal((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

    def _make_y_matrix(self, y):
        Y = (y.reshape(-1,1) == np.arange(self.n_classes).reshape(1,-1)).astype(np.float64)
        return np.where(Y==1, 1.0, -1.0)

    def decision_function(self, X):
        return X.dot(self.W) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return np.argmax(scores, axis=1)

    def fit(self, X, y, X_val=None, y_val=None):
        n = X.shape[0]
        idxs = np.arange(n)
        history = {'val_f1': []}
        best_f1 = -1
        no_improve = 0

        for epoch in range(self.epochs):
            RNG = np.random.default_rng(100 + epoch)
            RNG.shuffle(idxs)
            Xs, ys = X[idxs], y[idxs]

            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                xb = Xs[start:end]
                yb = ys[start:end]
                m = xb.shape[0]
                if m == 0: continue

                scores = xb.dot(self.W) + self.b
                Yb = self._make_y_matrix(yb)
                margins = Yb * scores
                mask = margins < 1
                R = mask.astype(np.float64) * Yb

                gradW = (self.W / self.C) - (xb.T @ R) / max(1, m)
                gradb = -np.sum(R, axis=0, keepdims=True) / max(1, m)

                lr_epoch = self.lr / (1 + 0.005 * epoch)
                self.W -= lr_epoch * gradW
                self.b -= lr_epoch * gradb

            if X_val is not None:
                y_pred = self.predict(X_val)
                val_f1 = f1_score(y_val, y_pred, average='weighted')
                history['val_f1'].append(val_f1)

                if self.verbose and ((epoch+1) % 20 == 0 or epoch == 0):
                    print(f"Epoch {epoch+1}/{self.epochs} val_f1={val_f1:.4f}")

                if epoch > 20:  
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        no_improve = 0
                    else:
                        no_improve += 1

                    if no_improve >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        return history

        return history


n_features = X_train_pca.shape[1]
n_classes = int(max(y_train.max(), y_val.max()) + 1)

svm = LinearSVM_OVR(
    n_features=n_features,
    n_classes=n_classes,
    lr=0.02,
    C=10.0,
    epochs=400,
    batch_size=256,
    patience=30,
    verbose=True
)

t0 = time.time()
hist_svm = svm.fit(X_train_pca, y_train, X_val=X_val_pca, y_val=y_val)
t1 = time.time()
svm_train_time=t1-t0

y_pred_svm = svm.predict(X_val_pca)
f1_svm = f1_score(y_val, y_pred_svm, average='weighted')
acc_svm = accuracy_score(y_val, y_pred_svm)

## Summary & save
results = [
    {'model': 'LogisticRegression', 'val_weighted_f1': f1_log, 'val_acc': acc_log, 'train_time_s': train_time},
    {'model': 'RandomForest', 'val_weighted_f1': f1_rf, 'val_acc': acc_rf, 'train_time_s': rf_train_time},
    {'model': 'OneVsRestGB', 'val_weighted_f1': f1_gb, 'val_acc': acc_gb, 'train_time_s': gb_train_time},
    {'model': 'SVM', 'val_weighted_f1': f1_svm, 'val_acc': acc_svm, 'train_time_s': svm_train_time}
]
summary = pd.DataFrame(results)
print(summary.sort_values('val_weighted_f1', ascending=False).to_string(index=False))

# save models and PCA for reproducibility
with open("pca.pkl", "wb") as f:
    pickle.dump(pca, f)
with open("logreg.pkl", "wb") as f:
    pickle.dump(logreg, f)
with open("rf.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("ovr_gb.pkl", "wb") as f:
    pickle.dump(ovr_gb, f)


#Tunning for Logistic Regression
print("Tunning for Logistic Regression")
print("tunning for lr")
for lr in [0.02, 0.01, 0.005]:
    model = SoftmaxLogisticRegression(n_features, n_classes, lr=lr)
    model.fit(X_train_pca, y_train)
    pred = model.predict(X_val_pca)
    print(lr, f1_score(y_val, pred, average='weighted'))
print("tunning for reg")
for reg in [1e-4, 1e-5, 1e-6]:
    model = SoftmaxLogisticRegression(n_features, n_classes, lr=0.02, reg=reg)
    model.fit(X_train_pca, y_train)
    pred = model.predict(X_val_pca)
    print(reg, f1_score(y_val, pred, average='weighted'))
print("tunning for batch size")
for bs in [128, 256, 512]:
    m = SoftmaxLogisticRegression(n_features, n_classes, lr=0.02, reg=1e-5, batch_size=bs)
    m.fit(X_train_pca, y_train)
    print(bs, f1_score(y_val, m.predict(X_val_pca), average='weighted'))

#Tunning for SVM
print("Tunning for SVM")
print("tunning for C")
for C in [10, 50, 100 , 300 , 500]:
    m = LinearSVM_OVR(n_features, n_classes, lr=0.02, C=C, epochs=400, batch_size=256)
    m.fit(X_train_pca, y_train)
    pred = m.predict(X_val_pca)
    print(C, f1_score(y_val, pred, average='weighted'))
print("tunning for lr")
for lr in [0.01, 0.02, 0.03]:
    m = LinearSVM_OVR(n_features, n_classes, lr=lr, C=300, epochs=400, batch_size=256)
    m.fit(X_train_pca, y_train)
    pred = m.predict(X_val_pca)
    print(lr, f1_score(y_val, pred, average='weighted'))











