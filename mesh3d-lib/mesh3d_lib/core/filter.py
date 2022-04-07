"""
Filtering methods aiming at removing outliers or groups of outliers from the point cloud.
"""

import pandas as pd
import numpy as np
import open3d as o3d
from cars.steps import points_cloud
import laspy


def radius_filtering_outliers_o3(cloud, radius, nb_points, serialize=True):
    """
    This method removes points that have few neighbors in a given sphere around them
    For each point, it computes the number of neighbors contained in a sphere of choosen radius,
    if this number is lower than nb_point, this point is deleted

    :param cloud: cloud point, it should be a pandas DataFrame or a numpy
    :param radius: defines the radius of the sphere that will be used for counting the neighbors
    :param nb_points: defines the minimm amount of points that the sphere should contain
    :return cloud: filtered pandas dataFrame cloud 
    """
    if isinstance(cloud, pd.DataFrame):
        data = cloud[["x", "y", "z"]]
        data = data.to_numpy()

    elif isinstance(cloud, np.ndarray):
        data = cloud

        if len(data.shape) != 2:
            raise ValueError(f"Data dimension is incorrect. It should be 2 dimensional. "
                             f"Found {len(data.shape)} dimensions.")
        if data.shape[1] != 3:
            raise ValueError("Data should be expressed as points along the rows and coordinates along the columns.")

    else:
        raise TypeError(f"Cloud is of an unknown type {type(cloud)}. It should either be a pandas DataFrame or a numpy " 
                        f"ndarray.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    cl,ind = pcd.remove_radius_outlier(10, radius)

    
    # get a final pandas cloud
    new_cloud = cloud.loc[ind]
    
    # serialize cloud in las
    if (serialize):
        serializeDataFrameToLAS(new_cloud, "/home/data/radiuso3dpyramide.las")

    return new_cloud


def small_components_filtering_outliers_cars(cloud, radius, nb_points, serialize=True):
    """
    This method removes small components that have not enough points inside
    For each point not yet processed, it computes the neighbors contained in a sphere of choosen radius, and the neighbors of neighbors ..
    until there are none left around. Those points are considered as processed and the identified cluster is added to a list
    For each cluster, if the number of points inside is lower than nb_point, this cluster is deleted

    :param cloud: cloud point, it should be a pandas DataFrame or a numpy
    :param radius: defines the radius of the sphere that will be used for counting the neighbors
    :param nb_points: defines the minimm amount of points that the sphere should contain
    :return cloud: filtered pandas dataFrame cloud 
    """
    if not (isinstance(cloud, pd.DataFrame) or isinstance(cloud, np.ndarray)):
        raise TypeError(f"Cloud is of an unknown type {type(cloud)}. It should either be a pandas DataFrame or a numpy " 
                        f"ndarray.")
    pos,_ = points_cloud.small_components_filtering(cloud,radius,nb_points)

    # serialize cloud in las
    if (serialize):
        serializeDataFrameToLAS(pos, "/home/data/radiuscarspyramidedefault.las")

    return pos

def statistical_filtering_outliers_cars(cloud, nb_neighbors, std_factor, serialize=True):
    """
    This methode removes points which have mean distances with their k nearest neighbors
    that are greater than a distance threshold (dist_thresh).

    This threshold is computed from the mean (mean_distances) and
    standard deviation (stddev_distances) of all the points mean distances
    with their k nearest neighbors:

        dist_thresh = mean_distances + std_factor * stddev_distances

    :param cloud: cloud point, it should be a pandas DataFrame or a numpy
    :param nb_neighbors: number of neighbors
    :param std_factor: multiplication factor to use to compute the distance threshold
    :return: filtered pandas dataFrame cloud
    """
    if not (isinstance(cloud, pd.DataFrame) or isinstance(cloud, np.ndarray)):
        raise TypeError(f"Cloud is of an unknown type {type(cloud)}. It should either be a pandas DataFrame or a numpy " 
                        f"ndarray.")

    pos,_ = points_cloud.statistical_outliers_filtering(cloud,100,3)

    # serialize cloud in las
    if (serialize):
        serializeDataFrameToLAS(pos, "/home/data/statscarspyramide.las")

    return pos

def statistical_filtering_outliers_o3d(cloud, nb_neighbors, std_factor, serialize=True):
    """
    This methode removes points which have mean distances with their k nearest neighbors
    that are greater than a distance threshold (dist_thresh).

    This threshold is computed from the mean (mean_distances) and
    standard deviation (stddev_distances) of all the points mean distances
    with their k nearest neighbors:

        dist_thresh = mean_distances + std_factor * stddev_distances

    :param cloud: cloud point, it should be a pandas DataFrame or a numpy
    :param nb_neighbors: number of neighbors
    :param std_factor: multiplication factor to use to compute the distance threshold
    :return: filtered pandas dataFrame cloud
    """
    if isinstance(cloud, pd.DataFrame):
        data = cloud[["x", "y", "z"]]
        data = data.to_numpy()

    elif isinstance(cloud, np.ndarray):
        data = cloud

        if len(data.shape) != 2:
            raise ValueError(f"Data dimension is incorrect. It should be 2 dimensional. "
                             f"Found {len(data.shape)} dimensions.")
        if data.shape[1] != 3:
            raise ValueError("Data should be expressed as points along the rows and coordinates along the columns.")

    else:
        raise TypeError(f"Cloud is of an unknown type {type(cloud)}. It should either be a pandas DataFrame or a numpy " 
                        f"ndarray.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    cl,ind = pcd.remove_statistical_outlier(100,3)

    # get a final pandas cloud
    new_cloud = cloud.loc[ind]

    # serialize cloud in las
    if (serialize):
        serializeDataFrameToLAS(new_cloud, "/home/data/statso3dpyramide.las")

    return new_cloud
    
    


def serializeDataFrameToLAS(cloud, pathout):
    """
    This method serializes a pandas DataFrame in .las
    A METTRE DANS TOOLS

    :param cloud: pandas DataFrame cloud
    :param pathout: which folder to write the file
    """
    header = laspy.LasHeader(point_format=8, version="1.4")
    # ~ print(dir(header))
    header.x_scale=1
    header.y_scale=1
    header.z_scale=1

    las = laspy.LasData(header)
    las.X = cloud["x"]
    las.Y = cloud["y"]
    las.Z = cloud["z"]
    las.red = cloud["clr0"]
    las.green = cloud["clr1"]
    las.blue = cloud["clr2"]
    las.nir = cloud["clr3"]

    las.write(pathout)
    
    
def main(df):
    xy = df[["x","y"]]
    densite = len(xy) / (xy.min()-xy.max()).prod()
    # radius and nb_points for cars method
    radius = np.sqrt(densite)
    nb_pts = densite*80
    print(nb_pts)
    print(radius)

    print("tot", len(df))
    cloudo3 = radius_filtering_outliers_o3(df, radius, nb_pts)
    # ~ cloudcars = small_components_filtering_outliers_cars(df, 3.0, 50)
    # ~ cloudcarstat = statistical_filtering_outliers_cars(df, 3, 10)
    # ~ cloudo3stat = statistical_filtering_outliers_o3d(df, 3, 10)
    # ~ print(len(cloudo3))
    # ~ print(len(cloudcars))
    # ~ print(len(cloudcarstat))
    # ~ print(len(cloudo3stat))

if __name__ == "__main__":
    fileName ='/home/code/stage/pyramide-points_color.pkl'
    df = pd.read_pickle(fileName)
    main(df)
