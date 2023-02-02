import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class DataGen:
    def __init__(self, n_clusters, n_points, scale=1, name="sym", seed=1234, as_pos=None):
        """Generate clusters of points around uniformly spaced points around origin in 2D, alternatively with asymmetric data

        Arguments:
            n_clusters -- Number of clusters
            n_points -- Number of points in each cluster

        Keyword Arguments:
            scale -- Scale centers of clusters uniformly (default: {1})
            name -- Name of cluster distribution, 
                options: 
                    "sym": uniform symmetric dist.
                    "as": asymmetric dist. with one symmetric cluster moved 
                    "as_extra": asymmetric dist. with a symmetric distribution with one extra cluster added
                    "parallel": parallel clusters
                (default: {"sym"})
            seed -- seed of random normal distribution (default: {1234})
            as_pos -- 2 length list of 2D coordinate of extra asymmetric cluster, None only if symmetric (default: {None})

        Raises:
            ValueError: If name is not one of the options
        """        """"""
        self.n_clusters = n_clusters
        self.n_points = n_points
        self.scale = scale
        self.seed = seed
        self.as_pos = as_pos
        if name == "sym":
            self.centers, self.data, self.labels = self.data_gen()
        elif name == "as":
            self.centers, self.data, self.labels = self.data_gen_as()
        elif name == "as_extra":
            self.centers, self.data, self.labels = self.data_gen_as_extra()
        elif name == "parallel":
            self.centers, self.data, self.labels = self.data_gen_parallel()
        elif name == "parallel_extra":
            self.centers, self.data, self.labels = self.data_gen_parallel_extra()
        else:
            raise ValueError("Invalid name")

    def data_gen(self):
        """Generate clusters of points around uniformly spaced points around origin in 2D

        Returns:
            [centers,data,labels]: Array of cluster centers, array of all 2D datapoints, array of all corresponding labels to datapoints
        """
        n_tot = self.n_clusters*self.n_points # Total number of datapoints
        centers = np.array([[self.scale*np.cos(t*2*np.pi/self.n_clusters),self.scale*np.sin(t*2*np.pi/self.n_clusters)] for t in range(self.n_clusters)]) # Generate centers uniformly around the origin
        data = np.zeros((n_tot,2))
        labels = np.zeros(n_tot)
        for i in range(self.n_clusters):
            np.random.seed(seed=self.seed)
            scatter = np.random.normal(loc=centers[i],scale=.1, size=(self.n_points,2))
            data[i*self.n_points:(i+1)*self.n_points] = scatter
            labels[i*self.n_points:(i+1)*self.n_points] = i
        return centers,data,labels

    # Asymmetric data self.n_clusters clusters
    def data_gen_as(self):
        """Generate clusters of points around uniformly spaced points around origin in 2D
        _as:    Asymmetric 3+1

        Returns:
            [centers,data,labels]: Array of cluster centers, array of all 2D datapoints, array of all corresponding labels to datapoints
        """
        n_tot = self.n_clusters*self.n_points # Total number of datapoints
        centers = np.array([[self.scale*np.cos(t*2*np.pi/self.n_clusters),self.scale*np.sin(t*2*np.pi/self.n_clusters)] for t in range(self.n_clusters-1)]) # Generate centers uniformly around the origin
        centers = np.concatenate((centers,[self.scale*self.as_pos]),axis=0)
        data = np.zeros((n_tot,2))
        labels = np.zeros(n_tot)
        for i in range(self.n_clusters):
            np.random.seed(seed=self.seed)
            scatter = np.random.normal(loc=centers[i],scale=.1, size=(self.n_points,2))
            data[i*self.n_points:(i+1)*self.n_points] = scatter
            labels[i*self.n_points:(i+1)*self.n_points] = i
        return centers,data,labels


    # Asymmetric data self.n_clusters+1
    def data_gen_as_extra(self):
        """Generate clusters of points around uniformly spaced points around origin in 2D
        as_extra: 4 + 1 asymmetric clusters

        Returns:
            [centers,data,labels]: Array of cluster centers, array of all 2D datapoints, array of all corresponding labels to datapoints
        """
        n_sym = self.n_clusters-1 # Number of symmetric clusters
        n_tot = self.n_clusters*self.n_points # Total number of datapoints
        centers = np.array([[self.scale*np.cos(t*2*np.pi/n_sym),self.scale*np.sin(t*2*np.pi/n_sym)] for t in range(n_sym)]) # Generate centers uniformly around the origin
        centers = np.concatenate((centers,[self.scale*self.as_pos]),axis=0)
        data = np.zeros((n_tot,2))
        labels = np.zeros(n_tot)
        for i in range(self.n_clusters):
            np.random.seed(seed=self.seed)
            scatter = np.random.normal(loc=centers[i],scale=.1, size=(self.n_points,2))
            data[i*self.n_points:(i+1)*self.n_points] = scatter
            labels[i*self.n_points:(i+1)*self.n_points] = i
        return centers,data,labels

    def data_gen_parallel(self):
        """Generate clusters of points around uniformly spaced points around origin in 2D
        parallel: 2 parallel clusters

        Returns:
            [centers,data,labels]: Array of cluster centers, array of all 2D datapoints, array of all corresponding labels to datapoints
        """
        n_tot = self.n_clusters*self.n_points # Total number of datapoints
        centers = np.array([[0,i*self.scale,] for i in range(1,self.n_clusters+1)]) # Generate centers along y-axis
        data = np.zeros((n_tot,2))
        labels = np.zeros(n_tot)
        for i in range(self.n_clusters):
            np.random.seed(seed=self.seed)
            scatter = np.random.normal(loc=centers[i],scale=.1, size=(self.n_points,2))
            data[i*self.n_points:(i+1)*self.n_points] = scatter
            labels[i*self.n_points:(i+1)*self.n_points] = i
        return centers,data,labels

    def data_gen_parallel_extra(self):
        """Generate clusters of points around uniformly spaced points around origin in 2D
        parallel: 2 parallel clusters + 1

        Returns:
            [centers,data,labels]: Array of cluster centers, array of all 2D datapoints, array of all corresponding labels to datapoints
        """
        n_tot = self.n_clusters*self.n_points # Total number of datapoints
        centers = np.array([[0,i*self.scale,] for i in range(1,self.n_clusters)]) # Generate centers along y-axis
        centers = np.concatenate((centers,[self.scale*self.as_pos]),axis=0)
        data = np.zeros((n_tot,2))
        labels = np.zeros(n_tot)
        for i in range(self.n_clusters):
            np.random.seed(seed=self.seed)
            scatter = np.random.normal(loc=centers[i],scale=.1, size=(self.n_points,2))
            data[i*self.n_points:(i+1)*self.n_points] = scatter
            labels[i*self.n_points:(i+1)*self.n_points] = i
        return centers,data,labels

    def plot(self):
        """Plot the dataset with the cluster centers as vectors"""
        colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))
        for col,center,i in zip(colors,self.centers,range(self.n_clusters)):
            plt.scatter(self.data[i*self.n_points:(i+1)*self.n_points,0],self.data[i*self.n_points:(i+1)*self.n_points,1],color=col)
            plt.arrow(0,0,center[0],center[1],length_includes_head=True,width=0.01,color=(0,0,0,0.5))

        plt.title(f"Dataset with mean as vectors, n={self.n_clusters}*{self.n_points}")
        plt.axis("equal")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.grid("on")
        plt.show()

    def dataset(self):
        return self.data,self.labels


# Independent functions:

def data_gen(n_cat,n_p,scale=1,seed = 1234):
    """Generate clusters of points around uniformly spaced points around origin in 2D

    Arguments:
        n_cat -- Number of categories
        n_p -- Number of points PER CATEGORY
        scale -- Scale of position of clusters

    Returns:
        [centers,data,labels]: Array of cluster centers, array of all 2D datapoints, array of all corresponding labels to datapoints
    """
    n_tot = n_cat*n_p # Total number of datapoints
    centers = np.array([[scale*np.cos(t*2*np.pi/n_cat),scale*np.sin(t*2*np.pi/n_cat)] for t in range(n_cat)]) # Generate centers uniformly around the origin
    data = np.zeros((n_tot,2))
    labels = np.zeros(n_tot)
    for i in range(n_cat):
        np.random.seed(seed=seed)
        scatter = np.random.normal(loc=centers[i],scale=.1, size=(n_p,2))
        data[i*n_p:(i+1)*n_p] = scatter
        labels[i*n_p:(i+1)*n_p] = i
    return centers,data,labels


# Asymmetric data n_cat clusters
def data_gen_as(n_cat,n_p,scale=1,as_pos=[1,-1],seed = 1234):
    """Generate clusters of points around uniformly spaced points around origin in 2D
    _as:    Asymmetric 3+1

    Arguments:
        n_cat -- Number of categories
        n_p -- Number of points PER CATEGORY
        scale -- Scale of position of clusters

    Returns:
        [centers,data,labels]: Array of cluster centers, array of all 2D datapoints, array of all corresponding labels to datapoints
    """
    n_tot = n_cat*n_p # Total number of datapoints
    centers = np.array([[scale*np.cos(t*2*np.pi/n_cat),scale*np.sin(t*2*np.pi/n_cat)] for t in range(n_cat-1)]) # Generate centers uniformly around the origin
    centers = np.concatenate((centers,[scale*as_pos]),axis=0)
    data = np.zeros((n_tot,2))
    labels = np.zeros(n_tot)
    for i in range(n_cat):
        np.random.seed(seed=seed)
        scatter = np.random.normal(loc=centers[i],scale=.1, size=(n_p,2))
        data[i*n_p:(i+1)*n_p] = scatter
        labels[i*n_p:(i+1)*n_p] = i
    return centers,data,labels


# Asymmetric data n_cat+1
def data_gen_as_extra(n_cat,n_p,scale=1,as_pos=[1,-1],seed = 1234):
    """Generate clusters of points around uniformly spaced points around origin in 2D
    as_extra: 4 + 1 asymmetric clusters

    Arguments:
        n_cat -- Number of categories
        n_p -- Number of points PER CATEGORY
        scale -- Scale of position of clusters

    Returns:
        [centers,data,labels]: Array of cluster centers, array of all 2D datapoints, array of all corresponding labels to datapoints
    """
    n_sym = n_cat-1 # Number of symmetric clusters
    n_tot = n_cat*n_p # Total number of datapoints
    centers = np.array([[scale*np.cos(t*2*np.pi/n_sym),scale*np.sin(t*2*np.pi/n_sym)] for t in range(n_sym)]) # Generate centers uniformly around the origin
    centers = np.concatenate((centers,[scale*as_pos]),axis=0)
    data = np.zeros((n_tot,2))
    labels = np.zeros(n_tot)
    for i in range(n_cat):
        np.random.seed(seed=seed)
        scatter = np.random.normal(loc=centers[i],scale=.1, size=(n_p,2))
        data[i*n_p:(i+1)*n_p] = scatter
        labels[i*n_p:(i+1)*n_p] = i
    return centers,data,labels


def plot_data(n_categories,n_p,data,centers):
    color_atlas = ["r","g","b","m","c","y","orange","brown","lime"]
    colors = []
    for i in range(n_categories):
        colors.append(color_atlas[i])

    for col,center,i in zip(colors,centers,range(n_categories)):
        plt.scatter(data[i*n_p:(i+1)*n_p,0],data[i*n_p:(i+1)*n_p,1],color=col)
        plt.arrow(0,0,center[0],center[1],length_includes_head=True,width=0.01,color=(0,0,0,0.5))

    plt.title(f"Dataset with mean as vectors, n={n_categories}*{n_p}")
    plt.axis("equal")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.grid("on")
    plt.show()