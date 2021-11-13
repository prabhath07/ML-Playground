# -----------------------------------------Imports --------------------------------------
import streamlit as st 
from sklearn.datasets import make_circles,make_moons,make_gaussian_quantiles,make_hastie_10_2
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as dtree
from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------Side Bar ----------------------------------------
st.sidebar.markdown("# Decision Tree Classifier")

dataset  = st.sidebar.selectbox('dataset',('Moons','Concentric Circles','Gaussian Quantiles','hastie_10_2'))

model = st.sidebar.selectbox('Model',('Logistic Regression','KNN','SVM','Decision Tree','Random Forest','Ensemble'))

#----------------Data Generation---------------------

def data(type ,samples=30,noise =0 ):
    X,y=(0,0)    
    if type=='Moons':
        X, y = make_moons(n_samples=samples, noise=noise, random_state=42)
    elif type=='Concentric Circles':
        X,y =make_circles(n_samples=samples, shuffle=True, noise=noise, random_state=None, factor=0.8)
    elif type=='Gaussian Quantiles':
        X,y = make_gaussian_quantiles(cov=noise*10+1, n_samples=samples, n_features=2, n_classes=2, shuffle=True)
    elif type=='hastie_10_2':
        X , y = make_hastie_10_2(n_samples=samples, random_state=None)       

    return X,y
    
#--------------------------------Mesh grid----------------------
def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array


#--------------------------------------------Model Prediction------------------------------------------------
labels,XX,YY=(0,0,0)
# def predict(model,X,y):
#     if model=='KNN':
#         clf = knn(n_neighbors=5,metric='minkowski')
#         clf.fit(X,y)
#         XX, YY, input_array = draw_meshgrid(X)
#         labels = clf.predict(input_array)
#         # pred = clf.predict()
    
#     return 0





#--------------------------------HOme -------------------------------------------

# samples = st.number_input('Number of Samples',min_value=10)
col1, col2= st.columns(2)
noise =0
# noise = st.slider("noise",min_value=0.00,max_value=1.00)
with col1:
    samples = st.number_input('Number of Samples',min_value=100)
if dataset=='Moons' or dataset =='Concentric Circles':
    with col2:
        noise = st.slider("noise",min_value=0.00,max_value=1.00,value=0.20)
    

    
    

#---plot-----
fig, ax = plt.subplots()
X,y = data(dataset,samples,noise)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
#-----------
orig = st.pyplot(fig)

# def run(fig,orig,clf,X_train,y_train,X_test,X,ax):
#     clf.fit(X_train,y_train)
#     XX, YY, input_array = draw_meshgrid(X)
#     labels = clf.predict(input_array)
#     y_pred =clf.predict(X_test)
#     # pred = clf.predict()
#     ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
#     plt.xlabel("Col1")
#     plt.ylabel("Col2")
#     orig.empty()
#     orig2 = st.pyplot(fig)
#     st.subheader("Accuracy is "+ str(round(accuracy_score(y_test, y_pred), 2)))
    
# -------------------------------------------------MOdel wise -----------------------------------
if model=='KNN':
    # orig.empty()
    neighbours = st.sidebar.number_input('N_Neighbours',min_value=1,max_value=10,value =5)
    P_value=st.sidebar.number_input('P Value',min_value=1,max_value=3,value =2)
    weights = st.sidebar.selectbox('Weights',('uniform','distance'))
    algo = st.sidebar.selectbox('Algorithm',('auto','dall-tree','kd_tree','brute'))
    clf = knn(n_neighbors=neighbours,metric='minkowski',p=P_value,algorithm=algo,weights=weights)
    if (st.sidebar.button('Run Model')):
        clf.fit(X_train,y_train)
        XX, YY, input_array = draw_meshgrid(X)
        labels = clf.predict(input_array)
        y_pred =clf.predict(X_test)
        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        plt.xlabel("Col1")
        plt.ylabel("Col2")
        orig.empty()
        orig2 = st.pyplot(fig)
        st.subheader("Accuracy of KNN model is "+ str(round(accuracy_score(y_test, y_pred), 2)))
        
elif model=='Decision Tree':
    criterion = st.sidebar.selectbox(    'Criterion',    ('gini', 'entropy'))

    splitter = st.sidebar.selectbox(    'Splitter',    ('best', 'random'))

    max_depth = int(st.sidebar.number_input('Max Depth',min_value=1,value=4))

    min_samples_split = st.sidebar.slider('Min Samples Split', 1, X_train.shape[0], 2,key=1234)

    min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1,key=1235)

    max_features = st.sidebar.slider('Max Features', 1, 2, 2,key=1236)

    max_leaf_nodes = int(st.sidebar.number_input('Max Leaf Nodes'))

    min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease')
    if max_depth == 0:
        max_depth = None

    if max_leaf_nodes == 0:
        max_leaf_nodes = None
    clf = dtree(criterion=criterion,splitter=splitter,max_depth=max_depth,random_state=42,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease)
    if (st.sidebar.button('Run Model')):
        clf.fit(X_train,y_train)
        XX, YY, input_array = draw_meshgrid(X)
        labels = clf.predict(input_array)
        y_pred =clf.predict(X_test)
        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        plt.xlabel("Col1")
        plt.ylabel("Col2")
        orig.empty()
        orig2 = st.pyplot(fig)
        st.subheader("Accuracy of Decision Tree is "+ str(round(accuracy_score(y_test, y_pred), 2)))
    
    
    
    



