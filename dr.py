import numpy as np
from scipy.spatial import distance_matrix
import math
import matplotlib.pyplot as plt


def mds(X=None, k=2, simi_X=None, important_paras=None):
    '''
        X: n x d
        important_paras: 1 * d
    '''
    if X is not None and important_paras is not None:
        X =  X * important_paras
    if X is not None:
        simi_X = np.dot(X, np.transpose(X))
           
    eval, evec = np.linalg.eig(simi_X)
    eval = eval.astype(np.float)
    eval[eval<10e-6] = 0
    evec = evec.astype(np.float)
    eval_diag = np.diag(eval)
    
    I = np.concatenate((np.diag([1,1]), np.zeros((2, simi_X.shape[0]-k))), axis=1)
    newX = np.dot(np.dot(I, np.sqrt(eval_diag)), np.transpose(evec))
    
    return np.transpose(newX)


def isomap(X, k=2, p=5):
    # compute D matrix 
    # where d_ij = || x_i - x_j||_2^2
    n = X.shape[0]
    theta = distance_matrix(X, X) ** 2
    # compute G based on p nearest neighbor
    index = np.argsort(theta)

    neatest_neighbor = [[] for _ in range(n)]
    for i, row in enumerate(index):
        for j, ele in enumerate(row):
            if ele != i and j<=p:
                neatest_neighbor[i].append(ele)

    print(index)
    print(neatest_neighbor)
    adj_matrix = np.full((n, n), np.inf)

    for i, row in enumerate(neatest_neighbor):
        for ele in row:
            adj_matrix[i][ele] = adj_matrix[ele][i] = theta[i][ele]

    for i in range(n):
        adj_matrix[i][i] = 0

    D = []
    for i in range(n):
        D.append(_dijkstra(adj_matrix, i))
    D = np.asarray(D)
    
    J = np.identity(n) - (1 / n)*(np.dot(np.ones((n,1)), np.ones((1,n))))
    S = -0.5 * np.dot(np.dot(J, D), J)
    new_X = mds(simi_X=S, k=k)

    return new_X


def _dijkstra(graph, source):
    '''
        graph: adjancency matrix 
    '''
    
    count = 0
    dist = []
    pre = [source for _ in range(graph.shape[0])]
    dist = [i for i in graph[source]]
    find = [False for _ in range(graph.shape[0])]

    find[source] = True
    v = source
    d = 0 
    while(count < graph.shape[0]):
        d = math.inf
        for i in range(graph.shape[0]):
            if (not find[i] and dist[i] < d):
                d = dist[i]
                v = i
        find[v] = True

        for i in range(graph.shape[0]):
            if not find[i]:
                d = dist[v] + graph[v][i]
                if d < dist[i]:
                    pre[i] = v
                    dist[i] = d
        
        count += 1

    return dist

def visualize(X, Y, name=None, title=None):
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    index = 0
    for point in X:

        plt.scatter(point[0], point[1], color=color[Y[index]-1])
        index += 1

    plt.title(title)
    if name is not None:
        plt.savefig(name)
    
    plt.show()


if __name__ == '__main__':
    pass


    
    

    





