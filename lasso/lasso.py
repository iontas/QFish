from numpy import array
from numpy.linalg import norm
import numpy as np
from sklearn.covariance import GraphicalLasso
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

sns.set_theme()
import warnings
warnings.filterwarnings("ignore")

class Lasso():
    def __init__(self,data):
        self.data = data
        
    """compute logreturns"""
    def logReturn(self):
        return np.log(self.data) - np.log(self.data.shift(1))

    """print stocks list"""
    def assetsList(self):
        return print(list(self.data.columns))
        
    def returnMean(self):
        return self.logReturn().mean()

    
    def returnStdv(self):
        return self.logReturn().std()
    
    """Normalized returns"""
    def normReturns(self):
        mu = self.returnMean()
        std = self.returnStdv()
        results = (self.logReturn()-mu)/std
        results = results.dropna()
        results = results.to_numpy()
        return results


    """Emperical Covariance from a given data"""
    def returnCovariance(self):
        return np.cov(self.normReturns())


    """input data as parameter for lasso fiting return estimated covariance"""
    def GraphicalLasso(self):
        return GraphicalLasso().fit(self.normReturns()).covariance_

    """heatmap seaborn"""
    def covarianceMap(self):
        plt.subplots(figsize=(16,10))
        sns.heatmap(self.GraphicalLasso(),cmap="YlGnBu", annot=True)
        plt.show()

    def CovarianceNetwork(self):
        G = nx.from_numpy_array(self.GraphicalLasso())
        nx.draw(G, with_labels=True, node_size=1500, node_color="skyblue", node_shape="s", alpha=0.5, linewidths=40)
        return plt.show()
    
    """plot kernel density for normal returns
    next release work on this feature
    def returnKde(self):
        ax = sns.displot(self.normReturns()[index], x="flipper_length_mm", kind="kde")
        plt.show()
    """
    def plotDistros(self):
        #for index, row in self.logReturn().iterrows():
        #    print(f'Index: {index}, row: {row.values}')
        plt.figure(figsize=(16, 10)) 
        for i in self.logReturn():
            #print(self.logReturn()[i])
            sns.distplot(self.logReturn()[i], hist=False)
        plt.title('Plot of Assets Log returns distributions')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.show()