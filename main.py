import numpy as np
from sklearn.decomposition import PCA
import dr

if __name__ == "__main__":
    data = []

    with open('zoo.data') as f:
        for line in f:
            line = line[:-1].split(',')[2:]
            data.append(line)


    data = np.array(data).astype(np.int8)

    X = data[:,:-1] # 101 * 15

    # center X
    x_centered = X - np.mean(X, axis=1).reshape(-1, 1)
    Y = data[:, -1]

    # PCA
    pca = PCA(n_components=2)
    p_com = pca.fit_transform(x_centered)
    #visualize and save images
    dr.visualize(p_com, Y, 'images/PCA.png', 'Visualization of PCA')

    # MDS
    # equal importance of each feature 
    new_x = dr.mds(X=x_centered)  
    dr.visualize(new_x, Y, 'images/MDS.png', 'Visualization of MDS')

    # assign different importance on each feature
    with open('importance_parameters.txt')  as f:
        important_paras = [i[:-1].split(' ') for i in f]
        important_paras = [[float(i) for i in para] for para in important_paras]
    for index, para in enumerate(important_paras):
        
        new_x = dr.mds(X=x_centered, important_paras=np.array(para).reshape(1,-1))
        dr.visualize(new_x, Y, 'images/MDS'+str(index)+'.png', 'Visualization of MDS')

    # isomap
    for p in [5,6,7,8,9,10,15,20,30]:
        new_X = dr.isomap(x_centered, p=p)
        dr.visualize(new_X, Y, 'images/Isomap-'+str(p)+'.png', 'Visualization of Isomap with p='+str(p))

