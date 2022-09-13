.. _denoise_point_cloud:

===================
Denoise point cloud
===================

Denoising a point cloud consists in retriveing the genuine surface given by the sparse representation.
Points are moved accordingly in order to fit a surface model.

Bilateral filtering
===================

Bilateral filtering is the most common way of achieving point cloud denoising.
Points are moved along their local normal (which represent the local surface direction) which is computed with the information available
(color, neighbours, etc.). The distance along the normal is computed as the mean distance to its N nearest neighbours
and its normal orientation according to the one of its neighbours.

It is supposed to respect the gradients in the point cloud while denoising flat surfaces.

As this approach is simple, it can be adapted to take as many parameters as desired. For example, one could use
a semantic information to apply different parameters for each class of object.
