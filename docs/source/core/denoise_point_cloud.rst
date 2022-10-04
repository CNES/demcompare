.. _denoise_point_cloud:

===================
Denoise point cloud
===================

Denoising a point cloud consists in retriveing the genuine surface given by the sparse representation.
Points are moved accordingly in order to fit a surface model.

Bilateral filtering
===================

    | *Action* : "denoise_pcd"
    | *Method* : "bilateral"

Bilateral filtering is the most common way of achieving point cloud denoising.
Points are moved along their local normal (which represents the local surface direction) which is computed with the information available
(color, neighbours, etc.). The distance along the normal is computed as the mean distance to its N nearest neighbours
and its normal orientation according to the one of its neighbours.

It is supposed to respect the gradients in the point cloud while denoising flat surfaces.

As this approach is simple, it can be adapted to take as many parameters as desired. For example, one could use
a semantic information to apply different parameters for each class of object.

This implementation was written from scratch according to the following paper: Digne, J., & Franchis, C.D. (2017). The Bilateral Filter for Point Clouds. Image Process. Line, 7, 278-287.
