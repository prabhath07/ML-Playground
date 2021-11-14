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
from sklearn.model_selection import cross_val_score
# ---------------------------------Side Bar ----------------------------------------
st.sidebar.markdown("# ML - Playground")

dataset  = st.sidebar.selectbox('dataset',('Moons','Concentric Circles','Gaussian Quantiles','hastie_10_2'))

model = st.sidebar.selectbox('Model',('---Select Algorithm---','Logistic Regression','KNN','SVM','Decision Tree','Random Forest','Ensemble'))

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





#--------------------------------HOme -------------------------------------------

# samples = st.number_input('Number of Samples',min_value=10)
col1, col2= st.columns(2)
noise =0
# noise = st.slider("noise",min_value=0.00,max_value=1.00)
with col1:
    samples = st.number_input('Number of Samples',min_value=30,value=100)
if dataset=='Moons' or dataset =='Concentric Circles':
    with col2:
        noise = st.slider("noise",min_value=0.00,max_value=1.00,value=0.20)
    

    
    

#---plot-----
fig, ax = plt.subplots()
X,y = data(dataset,samples,noise)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
#----------- Initial plot--------------------------------------------------------------
orig = st.pyplot(fig)

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
    
elif model=='Logistic Regression':
    penalty = st.sidebar.selectbox(    'Regularization',    ('l2', 'l1','elasticnet','none'))

    c_input = float(st.sidebar.number_input('C',value=1.0))

    solver = st.sidebar.selectbox(        'Solver',        ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')    )

    max_iter = int(st.sidebar.number_input('Max Iterations',value=100))

    multi_class = st.sidebar.selectbox(        'Multi Class',        ('auto', 'ovr', 'multinomial')    )

    l1_ratio = int(st.sidebar.number_input('l1 Ratio'))
    clf = LogisticRegression(penalty=penalty,C=c_input,solver=solver,max_iter=max_iter,multi_class=multi_class,l1_ratio=l1_ratio)
    if st.sidebar.button('Run Algorithm'):
        clf.fit(X_train,y_train)
        XX, YY, input_array = draw_meshgrid(X)
        labels = clf.predict(input_array)
        y_pred =clf.predict(X_test)
        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        plt.xlabel("Col1")
        plt.ylabel("Col2")
        orig.empty()
        orig2 = st.pyplot(fig)
        st.subheader("Accuracy for this LR Model is  " + str(round(accuracy_score(y_test, y_pred), 2)))

elif model=='Random Forest':
    n_estimators = int(st.sidebar.number_input('Num Estimators'))

    max_features = st.sidebar.selectbox(        'Max Features',        ('auto', 'sqrt','log2','manual')    )

    if max_features == 'manual':
        max_features = int(st.sidebar.number_input('Max Features'))

    bootstrap = st.sidebar.selectbox(        'Bootstrap',        ('True', 'False')    )
    if n_estimators == 0:
        n_estimators = 100
    max_samples = st.sidebar.slider('Max Samples', 1, X_train.shape[0], 1,key="1236")

    clf = RandomForestClassifier(n_estimators=n_estimators,random_state=42,bootstrap=bootstrap,max_samples=max_samples,max_features=max_features)
    if st.sidebar.button('Run Algorithm'):
        clf.fit(X_train,y_train)
        XX, YY, input_array = draw_meshgrid(X)
        labels = clf.predict(input_array)
        y_pred =clf.predict(X_test)
        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        plt.xlabel("Col1")
        plt.ylabel("Col2")
        orig.empty()
        orig2 = st.pyplot(fig)
        st.subheader("Accuracy for this LR Model is  " + str(round(accuracy_score(y_test, y_pred), 2)))

    
elif model=='Ensemble':
    ens = st.sidebar.selectbox('Ensemble Type',('Voting','Bagging'))
    if ens=='Voting':
        estimators = st.sidebar.multiselect('Estimators', 
        ['KNN','Logistic Regression', 'D-Tree','SVM','Random Forest'    ])

        voting_type = st.sidebar.radio(    "Voting Type",    (        'hard',        'soft',    ))
        
        es=[]
        for i in estimators:
            if i =='KNN':
                clf3 = knn(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
                es.append(('knn',clf3))
            elif i =='Logistic Regression':
                clf1 = LogisticRegression(penalty='l2',  dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
                es.append(('lr',clf1))
            elif i =='D-Tree':
                clf4 = dtree(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)
                es.append(('dtree',clf4))
            elif i =='Random Forest':
                clf5 = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
                es.append(('svm',clf5))
            elif i =='SVM':
                clf2 = SVC( C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)
                es.append(('svm',clf2))
        clf = VotingClassifier(estimators=es,voting=voting_type)
        if st.sidebar.button('Run Algorithm'):
            clf.fit(X_train,y_train)
            XX, YY, input_array = draw_meshgrid(X)
            labels = clf.predict(input_array)
            y_pred =clf.predict(X_test)
            ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
            plt.xlabel("Col1")
            plt.ylabel("Col2")
            orig.empty()
            orig2 = st.pyplot(fig)
            st.subheader("Accuracy for this Ensemble Model is  " + str(round(accuracy_score(y_test, y_pred), 2)))
        
    if ens =='Bagging':
        estimators = st.sidebar.multiselect('Estimators', 
        ['KNN','Logistic Regression', 'D-Tree','SVM','Random Forest'    ])
        n_estimators = int(st.sidebar.number_input('Enter number of estimators'))

        max_samples = st.sidebar.slider('Max Samples', 0, 375, 375,step=25)

        bootstrap_samples = st.sidebar.radio(            "Bootstrap Samples",            ('True', 'False')        )

        max_features = st.sidebar.slider('Max Features', 1, 2, 2,key=1234)

        bootstrap_features = st.sidebar.radio( "Bootstrap Features",   ('False', 'True'),       key=2345        )
        es=[]
        for i in estimators:
            if i =='KNN':
                clf3 = knn(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
                es.append(('knn',clf3))
            elif i =='Logistic Regression':
                clf1 = LogisticRegression(penalty='l2',  dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
                es.append(('lr',clf1))
            elif i =='D-Tree':
                clf4 = dtree(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)
                es.append(('dtree',clf4))
            elif i =='Random Forest':
                clf5 = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
                es.append(('svm',clf5))
            elif i =='SVM':
                clf2 = SVC( C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)
                es.append(('svm',clf2))
        clf = BaggingClassifier(es
        , n_estimators=n_estimators,
        max_samples=max_samples, bootstrap=bootstrap_samples,max_features=max_features,bootstrap_features=bootstrap_features, random_state=42)
        if st.sidebar.button('Run Algorithm'):
            clf.fit(X_train,y_train)
            XX, YY, input_array = draw_meshgrid(X)
            labels = clf.predict(input_array)
            y_pred =clf.predict(X_test)
            ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
            plt.xlabel("Col1")
            plt.ylabel("Col2")
            orig.empty()
            orig2 = st.pyplot(fig)
            st.subheader("Accuracy for this Ensemble Model is  " + str(round(accuracy_score(y_test, y_pred), 2)))
                
elif model=='SVM':
    gamma = st.sidebar.selectbox('Gamma',('auto','scale'))
    c = st.sidebar.number_input('C',value = 1.0)
    kernel = st.sidebar.selectbox('Kernel',('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'))
    degree=st.sidebar.number_input('Degree',min_value=1.00,value = 3.00)
    clf = SVC( C=c, kernel=kernel, degree=degree, gamma=gamma, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)
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
        st.subheader("Accuracy of SVM model is "+ str(round(accuracy_score(y_test, y_pred), 2)))
    

