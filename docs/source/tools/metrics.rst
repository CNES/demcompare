.. _metrics:

=======
Metrics
=======

In order to compare two data or quantify the contribution of the processings, some general metrics are implemented
as well as a visualisation tool.

The tools developed compare two point clouds, not the faces of meshes (if a mesh is input, its vertices are used for comparison).
Faces comparison is not implemented for now. `Metro <http://vcg.isti.cnr.it/vcglib/metro.html>`_ tool based on the VCG Lib can be used to extract mesh metrics. However, the visuals generation has a bug (reported
in a github issue).

Point Cloud Quantification
==========================

Quantify the contribution of processings is an important subject. However, the right way to compare two point clouds
or meshes is not obvious. To our knowledge, no simple metric allows to capture all the complexity of 3D information.

Two ways of computing metrics are implemented:

* with a **point to point** approach : For each point in the cloud, its nearest neighbour (in the euclidean distance sense) is retrieved and the distance between the points is used to compute the statistics.
* with a **point to surface** approach : For each point in the cloud, its nearest neighbour (in the euclidean distance sense) is retrieved and projected onto the line defined by the point in the cloud and its local normal. The projected distance is then used to compute the statistics.

.. note::

    Point cloud metrics are not symmetric. Comparing point cloud A to point cloud B does not give the same result as
    comparing point cloud B to point cloud A. Indeed, let's take the example of a random point cloud A and a point
    cloud B being the exact same point cloud but with some points removed.

    * **When comparing A to B** : Metrics will give non zero values.
    * **When comparing B to A** : All the asymmetric metrics will be zero since all the points of B equal a point in A.


The following metrics are implemented:

* Mean Squared Distance (MSD, or MSE)
* Root Mean Squared Distance (RMSD, or RMSE)
* Mean Distance
* Median Distance
* Hausdorff Asymmetric Distance
* Hausdorff Symmetric Distance
* Chamfer Distance

There are computed both ways (A > B and B > A) and are saved in a CSV file.

Source : Lang Zhou, Guoxing Sun, Yong Li, Weiqing Li, Zhiyong Su, Point cloud denoising review: from classical to deep
learning-based approaches, Graphical Models, Volume 121, 2022, 101140, ISSN 1524-0703,
https://doi.org/10.1016/j.gmod.2022.101140.

Point Cloud Qualification
==========================

Since quantitative metrics cannot capture all the information needed to evaluate a point cloud, we propose qualitative
analysis by providing visuals. In the same fashion as paper authors do, we save two point clouds (one for A > B
comparison, and the other for B > A comparison) with a scalar field corresponding to the distance (point to point,
or point to surface) between the point and its nearest neighbour. Thus, by configurating a color map, one can easily
see the regions with the highest number of differences.
