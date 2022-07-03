from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
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
    Times = []
    cm=np.zeros((10,10))
    Precision=[]
    Recall=[]
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
        cm = cm + confusion_matrix(y_test,y_pred,labels=[u for u in range(10)])
        other_metrics=precision_recall_fscore_support(y_test,y_pred,labels=[u for u in range(10)])
        Precision.append(other_metrics[0])
        Recall.append(other_metrics[1])
        F_Score.append(other_metrics[2])

    Accuracities = np.asarray(Accuracities)
    Times = np.asarray(Times)
    return [np.mean(Accuracities),np.mean(Times),np.mean(Precision),np.mean(Recall),np.mean(F_Score),cm]

k = 5
K = 6
alfa = 10
results = tests_KFolds(k, K, alfa, 'data/fashion-mnist_all.csv')
print(results)
with open( 'log.txt', 'w') as file:
    print(results, file = file)
