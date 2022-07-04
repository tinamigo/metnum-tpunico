from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

#LEER https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9

# k de kNN, K de Kfold
def tests_KFolds(k, K, alfa, trainpath):

    df_train = pd.read_csv(trainpath)

    X = df_train[df_train.columns[1:]].values
    y = df_train["label"].values.reshape(-1, 1)

    #aplico PCA
    y = y.ravel()
    pca = PCA(n_components=alfa)
    X = pca.fit_transform(X,y)

    # separo con k-fold
    kfold = KFold(n_splits=K)
    kfold.get_n_splits(X)

    #Para cada fold se hara un entrenamiento, prediccion y se guardara el accuracity y el tiempo que dio cada uno
    Accuracities = []
    F_Score=[]

    for train_index, test_index in kfold.split(X):
        # separo train y test para este fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        # armo y entreno un clasif knn
        kNN = KNeighborsClassifier(k)
        kNN.fit(X_train, y_train)

        # predigo
        y_pred = kNN.predict(X_test)

        #Se ve el accuracity de lo predicho
        acc = accuracy_score(y_test, y_pred)
        Accuracities.append(acc)
        fscore = f1_score(y_test,y_pred,labels=[u for u in range(10)], average= 'macro')
        F_Score.append(fscore)

    Accuracities = np.asarray(Accuracities)
    return np.mean(Accuracities),np.mean(F_Score)

K = 6
results = pd.DataFrame(columns = ['k','alfa','acc','fscore'])
for k in range(5,7):
    print("k: ", k)
    for i in range(2,15):
        alfa = i*10
        print("  Alfa: ", alfa)
        acc, fscore = tests_KFolds(k, K, alfa, 'data/fashion-mnist_all.csv')
        print("-acc: ", acc)
        new_row = {'k':k,'alfa':alfa,'acc':acc,'fscore':fscore}
        results = results.append(new_row,ignore_index=True)

print(results)
results.to_csv('results2.csv')

#max 0.8587999883405649
