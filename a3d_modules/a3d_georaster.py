#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

"""
a3d_georaster contains classes to deal with georeferenced raster

A3DGeoRaster aims at simplifying the use of georeferenced raster.
It provides access to :
    cls.ds :   the GDAL handle to the dataset, which provides access to all GDAL functions :
               see http://www.gdal.org/classGDALDataset.html

    cls.srs :  an OGR Spatial Reference object representation of the dataset.
               see http://www.gdal.org/classOGRSpatialReference.html

    cls.proj : a pyproj coordinate conversion function between the dataset coordinate system and lat/lon.

    cls.footprint : tuple of the corners of the dataset in native coordinate system, as (left,right,bottom,top).

    cls.trans : geoTransform tuple of the dataset.

    cls.rpc : rpc is provided

    cls.raster : numpy array for accessing the data, m*n for SingleBandRaster, m*n*bands for MultiBand.


A3DDEMRaster aims at simplifying the use of DEM georeferenced raster.
It adds DEM specific methods and works with geoid reference
"""

import numpy as np
from osgeo import gdal, osr, ogr
from astropy import units as u
from scipy.ndimage import filters
try:
    import pyproj
except ImportError:
    import mpl_toolkits.basemap.pyproj as pyproj


class A3DGeoRaster(object):
    """
    A Geo raster dataset initialized from a file path to a single band raster dataset of GDAL understood type.

    :arg ds_file : filepath and name to raster (is updated if raster is saved to another location)
    :arg ds : GDAL dataset handle
    :arg footprint : footprint of raster in geo coordinates of the form (left, right, bottom, top)
    :arg srs : OSR SpatialReference object
    :arg proj : pyproj conversion object raster coordinates<->lat/lon
    :arg nx, ny : x and y loaded raster size
    :arg xres, yres: pixel resolution
    :arg x0, y0 : the offsets in x and y of the loaded raster area
    :arg r: raster as numpy array
    :arg dtype: data type

    Example:
    >>> a3d_georaster.A3DGeoRaster('myfile.tif',load_data=True|False)

    Example when using a roi :
    >>> from a3d_modules.a3d_georaster import A3DGeoRaster
    >>> geo=A3DGeoRaster('mygeoraster.tif', load_data=True)
    >>> geo.r=geo.r[2000:2010,1000:1010]
    >>> geo2=A3DGeoRaster('mygeoraster.tif', load_data={'x':1000,'y':2000,'w':10,'h':10})
    >>> import numpy as np
    >>> np.count_nonzero(geo.r-geo2.r)
    0
    >>> geo.nx == geo2.nx
    False
    >>> geo=A3DGeoRaster('ref.tif')
    >>> geo.r=geo._load_raster_subset_from_roi({'x':1000,'y':2000,'w':10,'h':10}, update_info=True)
    >>> np.count_nonzero(geo.r-geo2.r)
    0
    >>> geo.nx == geo2.nx
    True
    """
    # Filepath and name
    ds_file = None
    # GDAL handle to the dataset
    ds = None
    # GeoTransform
    trans = None
    # Footprint of raster
    footprint = None
    # SRS
    srs = None
    # pyproj Projection
    proj = None
    # Raster size
    nx = None
    ny = None
    # Pixel size
    xres = None
    yres = None
    # Numpy array of band data
    r = None
    # Band datatype
    dtype = None
    # No data value
    nodata = None
    # RPC
    rpc = None
    # Band
    band = 1
    # plani unit (supposed to always be meter so far) TODO
    plani_unit = u.meter

    def __del__(self):
        """
        Close gdal dataset
        """
        self.ds = None

    def __init__(self,ds_filename,
                 spatial_ref_sys=None,geo_transform=None,rpc=None,nodata=None,band=1,load_data=True,latlon=False):

        # Do basic dataset loading - set up georeferencing))
        self._import_ds(ds_filename, spatial_ref_sys=spatial_ref_sys, geo_transform=geo_transform)

        # Import band datatype
        band_tmp = self.ds.GetRasterBand(band)
        self.dtype = gdal.GetDataTypeName(band_tmp.DataType)

        # Load entire image
        if load_data is True:
            self.r = self._load_raster(band)
        # Or load just a subset region
        elif isinstance(load_data, tuple) or isinstance(load_data, dict):
            if not isinstance(load_data, dict):
                # if load_data is a tuple or a list then coordinates are georeferenced
                load_data = self.footprint_to_roi(load_data, latlon=latlon)
            self.r = self._load_raster_subset_from_roi(load_data, band=band, update_info=True)
        elif load_data is False:
            return
        else:
            print('Warning : load_data argument not understood. No data loaded.')

        # Deal with no data by setting no data to np.nan
        self.r = np.float32(self.r)
        meta_nodata = self.ds.GetRasterBand(1).GetNoDataValue()
        if meta_nodata is not None:
            # we to explicitly write 'is not None' because 0.0 can be a no data value and the if won't take it otherwise
            self.r[self.r == meta_nodata] = np.nan
            self.nodata = meta_nodata
        else:
            # if no meta_nodata we set one to -32768
            self.nodata = -32768
            self.ds.GetRasterBand(1).SetNoDataValue(self.nodata)
            self.r[self.r == self.nodata] = np.nan
        if nodata is not None:
            # we change the nodata value as wished
            self.ds.GetRasterBand(1).SetNoDataValue(nodata)
            self.r[self.r == nodata] = np.nan
            self.nodata = nodata

    @classmethod
    def from_raster(cls, raster, geo_transform, proj4, gdal_dtype=gdal.GDT_Float32, nodata=None):
        """
        Create A3DGeoRaster from raster and geo information

        :param raster: 2D numpy array
        :param geo_transform: gdal geo transform
        :param proj4: spatial reference system as proj4 string format
        :param gdal_dtype:
        :param nodata:
        :return: A3DGeoRaster
        """
        #
        # if len(raster.shape) > 2:
        #     nbands = raster.shape[2]
        # else:
        nbands = 1

        # Create a GDAL memory raster to hold the input array
        mem_drv = gdal.GetDriverByName('MEM')
        source_ds = mem_drv.Create('', raster.shape[1], raster.shape[0], nbands, gdal_dtype)

        # Set geo-referencing
        source_ds.SetGeoTransform(geo_transform)
        srs = osr.SpatialReference()
        srs.ImportFromProj4(proj4)
        source_ds.SetProjection(srs.ExportToWkt())

        # Write input array to the GDAL memory raster
        for b in range(0,nbands):
            if nbands > 1:
                r = raster[:,:,b]
            else:
                r = raster
            source_ds.GetRasterBand(b+1).WriteArray(r)
            if nodata != None:
                source_ds.GetRasterBand(b+1).SetNoDataValue(nodata)

        # Return a fully loaded georaster instance instantiated by the GDAL raster
        return cls(source_ds)

    def _import_ds(self, ds_filename, spatial_ref_sys=None, geo_transform=None):
        """
        Import dataset metadata and store them for later use

        :param ds_filename: dataset name or just gdal dataset
        :param spatial_ref_sys: gdal srs
        :param geo_transform: gdal trans
        """

        # Open file
        if isinstance(ds_filename, str):
            self.ds_file = ds_filename
            self.ds = gdal.Open(ds_filename, 0)
        # Or import GDAL Dataset from memory
        elif isinstance(ds_filename, gdal.Dataset):
            self.ds = ds_filename
            self.ds_file = ds_filename.GetDescription()

        # Check that some georeferencing information is available
        if self.ds.GetProjection() == '' and spatial_ref_sys is None:
            print('WARNING: No geo information. This might not be a geo raster')

        # Check user defined geo info
        if (spatial_ref_sys is None and geo_transform is not None) or \
                (spatial_ref_sys is not None and geo_transform is None):
            raise NameError('ERROR: You must set both spatial_ref_sys and geo_transform.')

        if spatial_ref_sys is None:
            self.trans = self.ds.GetGeoTransform()
            self.srs = osr.SpatialReference()
            self.srs.ImportFromWkt(self.ds.GetProjection())
        else:
            # GeoTransform
            self.trans = geo_transform
            # Spatial Reference System
            self.srs = spatial_ref_sys

        # Create footprint tuple in native dataset coordinates
        self.footprint = ( self.trans[0],
                           self.trans[0] + self.ds.RasterXSize * self.trans[1],
                           self.trans[3] + self.ds.RasterYSize * self.trans[5],
                           self.trans[3] )

        # Pixel size
        self.xres = self.trans[1]
        self.yres = self.trans[5]

        # Raster size
        self.nx = self.ds.RasterXSize
        self.ny = self.ds.RasterYSize

        # Load projection if there is one
        if self.srs.IsProjected():
            self.proj = pyproj.Proj(self.srs.ExportToProj4())

    def _load_raster(self, band=1):
        """
        Load raster (single band)
        
        :param band: int, band id 
        :return: numpy array (x,y)
        """
        band = int(band)

        return self.ds.GetRasterBand(band).ReadAsArray()

    def _load_raster_subset_from_roi(self, roi, band=1, update_info=False):
        """
        Load raster subset defined by roi with pixels coordinates

        :param roi: {'x':,'y':,'w':,'h'}
        :param band: int, band to read from
        :param update_info: boolean, set to true if current raster information are to be modified (if self.r = return)
        :return: array associated with the roi
        """

        xpx1 = roi['x']
        x_offset = roi['w']
        ypx1 = roi['y']
        y_offset = roi['h']

        # In special case of being called to read a single point, offset 1 px
        if x_offset == 0: x_offset = 1
        if y_offset == 0: y_offset = 1

        # Read array and return
        arr = self.ds.GetRasterBand(band).ReadAsArray( int(xpx1), int(ypx1), int(x_offset), int(y_offset))

        # Update image size
        # (top left x, w-e px res, 0, top left y, 0, n-s px res)
        trans = self.ds.GetGeoTransform()
        left = trans[0] + xpx1 * trans[1]
        top = trans[3] + ypx1 * trans[5]
        subset_footprint = (left, left + x_offset * trans[1], top + y_offset * trans[5], top)
        if update_info is True:
            self.nx, self.ny = int(x_offset), int(y_offset)  # arr.shape
            self.x0 = int(xpx1)
            self.y0 = int(ypx1)
            self.footprint = subset_footprint
            self.trans = (left, trans[1], 0, top, 0, trans[5])

        return arr
            
    def footprint_to_roi(self, footprint, latlon=False):
        """
        Convert footprint to a3d roi in pixel coordinates.

        :param footprint: (left, right, bottom, top)
        :param latlon: boolean, default False. Set as True if footprint in lat/lon.
        :return: roi associated
        """

        left = footprint[0]
        right = footprint[1]
        bottom = footprint[2]
        top = footprint[3]

        # Unlike the bounds tuple, which specifies bottom left and top right
        # coordinates, here we need top left and bottom right for the numpy
        # readAsArray implementation.
        xpx1, ypx1 = self.coord_to_px(left, bottom, latlon=latlon)
        xpx2, ypx2 = self.coord_to_px(right, top, latlon=latlon)

        if xpx1 > xpx2:
            xpx1, xpx2 = xpx2, xpx1
        if ypx1 > ypx2:
            ypx1, ypx2 = ypx2, ypx1

        return {'x': xpx1, 'y': ypx1, 'w': xpx2-xpx1, 'h': ypx2-ypx1}

    def roi_to_footprint(self, roi):
        """
        Convert a3d roi to footprint in coordinates.

        :param roi: {'x':,'y':,'w':,'h'}
        :return: footprint tuple associated
        """

        left = roi['x']
        right = left + roi['w']
        top = roi['y']
        bottom = top + roi['h']
        left, top = self.px_to_coord(left, top)
        right, bottom = self.px_to_coord(right, bottom)
        return left, right, bottom, top

    def px_to_coord(self, xP=None, yP=None, latlon=False):
        """
        Convert xP,yP pixel coordinates to raster into coordinates.
        Note that xP,yP = (0,0) does not refer to left, top geo raster corner but to left +0.5, top +0.5

        >>> from a3d_modules.a3d_georaster import A3DGeoRaster
        >>> geo = A3DGeoRaster('myGeoRaster.tif')
        >>> xGeo, yGeo = geo.px_to_coord(-0.5,-0.5)
        >>> xGeo == geo.trans[0]
        >>> True

        :param xP : int, x pixel coordinate to convert.
        :param yP : int, y pixel coordinate to convert.
        :param latlon : boolean, default False. Set as True if output bounds in lat/lon.
        :return: x,y which may be either in native coordinate system of raster or lat/lon.
        """
        if np.size(xP) != np.size(yP):
            print("Xpixels and Ypixels must have the same size")
            return 1

        if (xP is None) & (yP is None):
            xP = np.arange(self.nx)
            yP = np.arange(self.ny)
            xP, yP = np.meshgrid(xP, yP)
        else:
            Xpixels = np.array(xP)
            Ypixels = np.array(yP)

        # coordinates are at centre-cell, therefore the +0.5
        trans = self.trans
        Xgeo = trans[0] + (xP + 0.5) * trans[1] + (yP + 0.5) * trans[2]
        Ygeo = trans[3] + (xP + 0.5) * trans[4] + (yP + 0.5) * trans[5]

        if latlon is True:
            Xgeo, Ygeo = self.proj(Xgeo, Ygeo, inverse=True)

        return (Xgeo, Ygeo)

    def coord_to_px(self, x, y, latlon=False, rounded=True, check_valid=True):
        """
        Convert x,y geo coord to pixel coord

        :param x:
        :param y:
        :param latlon: boolean, set to true if latlon coord
        :param rounded: boolean, set to true if round required
        :param check_valid: boolean, set to true if one wants to check px correspondance
        :return: xP, yP
        """

        # Convert coordinates to map system if provided in lat/lon and image
        # is projected (rather than geographic)
        if latlon is True and self.proj is not None:
            x, y = self.proj(x, y)

        # Shift to the centre of the pixel
        x = np.array(x - self.xres / 2)
        y = np.array(y - self.yres / 2)

        g0, g1, g2, g3, g4, g5 = self.trans
        if g2 == 0:
            xPixel = (x - g0) / float(g1)
            yPixel = (y - g3 - xPixel * g4) / float(g5)
        else:
            xPixel = (y * g2 - x * g5 + g0 * g5 - g2 * g3) / float(g2 * g4 - g1 * g5)
            yPixel = (x - g0 - xPixel * g1) / float(g2)

        # Round if required
        if rounded is True:
            xPixel = np.round(xPixel)
            yPixel = np.round(yPixel)

        if check_valid is False:
            return xPixel, yPixel

        # Check that pixel location is not outside image dimensions
        nx = self.ds.RasterXSize
        ny = self.ds.RasterYSize

        xPixel_new = np.copy(xPixel)
        yPixel_new = np.copy(yPixel)
        xPixel_new = np.fmin(xPixel_new, nx)
        yPixel_new = np.fmin(yPixel_new, ny)
        xPixel_new = np.fmax(xPixel_new, 0)
        yPixel_new = np.fmax(yPixel_new, 0)

        if np.any(xPixel_new != xPixel) or np.any(yPixel_new != yPixel):
            print("WRANING: some points are out of domain for file")

        return xPixel_new, yPixel_new

    def coord_to_coord(self, x, y, target_srs):
        """
        Convert x,y geo coord to x,y, geo coord of another spatial reference system

        :param x:
        :param y:
        :param target_srs: srs, target srs
        :return: x, y in target srs
        """

        # Create Coord Transform object from source srs to target one
        oCT = osr.CoordinateTransformation(self.srs, target_srs)

        # Perform transformation
        x, y, z = oCT.TransformPoint(x, y)
        return x, y

    def reproject(self, target_srs, nx, ny, xmin, ymax, xres, yres,
                  dtype=gdal.GDT_Float32, nodata=None,
                  interp_type=gdal.GRA_Bilinear, progress=False):
        """
        Reproject a dataset into another georeferenced system. This can be useful to resample a georaster.
        This method relies on gdal.ReprojectImage

        :param target_srs: gdal srs format for destination
        :param nx: int, width
        :param ny: int, height
        :param xmin: int, min column coord
        :param ymax: int, max row coord
        :param xres:
        :param yres:
        :param dtype: gdal data type
        :param nodata: nodata value for the ones generated by the reproject method
        :param interp_type: gdal interpolator
        :param progress: boolean, set to True to display a progress bar
        :return: a new A3DGeoRaster object

        :Example:

        >>> from a3d_modules.a3d_georaster import A3DGeoRaster
        >>> georaster=A3DGeoRaster('mygeoRaster.tif')
        >>> coord = georaster.px_to_coord(40.3,40.7)
        >>> reproj_georaster = georaster.reproject(georaster.srs, 100, georaster.ny, coord[0], coord[1], georaster.xres, georaster.yres*2)
        >>> reproj_georaster.nx
        100
        >>> reproj_georaster.footprint[0] = coord[0]
        """

        # Create an in-memory raster
        mem_drv = gdal.GetDriverByName('MEM')
        target_ds = mem_drv.Create('', nx, ny, 1, dtype)

        # Set the new geotransform
        new_geo = (xmin, xres, 0, ymax, 0, yres)
        target_ds.SetGeoTransform(new_geo)
        target_ds.SetProjection(target_srs.ExportToWkt())

        # Set the nodata value
        if nodata is not None:
            for b in range(1, self.ds.RasterCount + 1):
                inBand = self.ds.GetRasterBand(b)
                inBand.SetNoDataValue(nodata)

        # Perform the projection / resampling
        if progress is True:
            res = gdal.ReprojectImage(self.ds, target_ds, None, None,
                                      interp_type, 0.0, 0.0, gdal.TermProgress)
        else:
            res = gdal.ReprojectImage(self.ds, target_ds, None, None,
                                      interp_type, 0.0, 0.0, None)

        # Load data
        new_raster = self.__class__(target_ds)
        new_raster.r[new_raster.r == 0] = nodata

        return new_raster

    def crop(self, extract_win, latlon=False):
        """
        Return cropped A3DGeoRaster

        :param extract_win: {'x':,'y':,'w':,'h':} roi or (left, right, bottom, top) footprint
        :param latlon: boolean, set to True footprint coordinates are lat/lont
        :return: a cropped new A3DGeoRaster object
        """

        return self.__class__(self.ds, load_data=extract_win, latlon=latlon, nodata=self.nodata, rpc=self.rpc, band=self.band)

    def geo_translate(self, x_off, y_off, system='coord', latlon=False):
        """
        Translate the geoloc of a georaster without altering the raster itself. This can be useful whenever the dataset
        trans is wrong according to a coregistration algorithm and a reference georaster.

        :param x_off:
        :param y_off:
        :param system: 'coord' or 'pixel' is the coordinates system
        :param latlon: boolean, set to True if system is 'coord' and coordinates are lat/lont

        :example:

        >>> from a3d_modules.a3d_georaster import A3DGeoRaster
        >>> dem1=A3DGeoRaster('s2p.tif',nodata=-32768)
        >>> x_off,y_off=dem1.px_to_coord(0.44,0.66)
        >>> dem1.geo_translate(x_off,y_off)
        >>> x_off - dem1.trans[0]
        0.0
        >>> dem1=A3DGeoRaster('s2p.tif',nodata=-32768)
        >>> dem1.geo_translate(0.44,0.66,system='pixel')
        >>> x_off - dem1.trans[0]
        0.0
        """

        if system == 'pixel':
            x_off, y_off = self.px_to_coord(x_off, y_off, latlon=False)
        else:
            if latlon is True:
                x_off, y_off = self.proj(x_off, y_off)

        # update trans
        trans = (x_off, self.trans[1], self.trans[2], y_off, self.trans[4], self.trans[5])
        target_ds = write_geotiff('', self.r, trans, wkt=self.srs.ExportToWkt(), nodata=self.nodata)
        new_raster = self.__class__(target_ds)

        return new_raster

    def biggest_common_footprint(self, other_geo_raster):
        """
        Get biggest common footprint between self and filename dataset

        :param other_geo_raster: A3DGeoRaster, to intersect footprint with
        :return: (left, right, bottom, top) coordinates
        """

        # Build polygons
        # - self
        left, right, bottom, top = self.footprint
        wkt = "POLYGON ((%f %f, %f %f, %f %f, %f %f, %f %f))" % (left, bottom, left, top, right, top, right, bottom,
                                                                 left, bottom)
        poly1 = ogr.CreateGeometryFromWkt(wkt)
        # - other dataset
        left, right, bottom, top = other_geo_raster.footprint
        # - eventually translate other dataset coordinates to self osr
        left, top = other_geo_raster.coord_to_coord(left, top, target_srs=self.srs)
        right, bottom= other_geo_raster.coord_to_coord(right, bottom, target_srs=self.srs)
        wkt = "POLYGON ((%f %f, %f %f, %f %f, %f %f, %f %f))" % (left, bottom, left, top, right, top, right, bottom,
                                                                 left, bottom)
        poly2 = ogr.CreateGeometryFromWkt(wkt)

        # Get back biggest common footprint
        intersect = poly1.Intersection(poly2)
        footprint = intersect.GetEnvelope()

        # check that intersection is not void
        if intersect.GetArea() == 0:
            print('Warning: Intersection is void')
            return 0
        else:
            return footprint

    def _get_orthodromic_distance(self, lon1, lat1, lon2, lat2):
        """
        Get Orthodromic distance from two (lat,lon) coordinates

        :param lon1:
        :param lat1:
        :param lon2:
        :param lat2:
        :return: orthodromic distance
        """
        Re = 6378137.0  # WGS-84 equatorial radius in km
        return Re * np.arccos(
            np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) * np.cos((lon2 - lon1) * np.pi / 180) + np.sin(
                lat1 * np.pi / 180) * np.sin(lat2 * np.pi / 180))

    def save_geotiff(self, filename, dtype=gdal.GDT_Float32):
        """
        Save georaster into file system.

        :param filename: output filename
        :param dtype: gdal type
        """
        # deal with unicode input
        filename = str(filename)

        # replace nan by nodata before writing
        self.r[np.where(np.isnan(self.r))] = self.nodata

        dst_ds = write_geotiff(filename, self.r, self.trans, wkt=self.srs.ExportToWkt(), dtype=dtype, nodata=self.nodata)

        # update the dataset in case we still want to work on it
        if dst_ds is True:
            self._import_ds(filename)
        else:
            self._import_ds(dst_ds)

        # replace nodata by nan after writing
        self.r[self.r == self.nodata] = np.nan


class A3DDEMRaster(A3DGeoRaster):
    # Z Unit
    zunit = None

    def __init__(self, ds_filename, band=1, ref='WGS84', nodata=None, zunit='m', load_data=True, latlon=False, rpc=None):
        super(A3DDEMRaster, self).__init__(ds_filename, nodata=nodata, load_data=load_data, latlon=latlon, band=band)

        # Convert to float32 for future computations
        self.r = np.float32(self.r)

        # Convert to meter (so all A3DDEMRaster have meter as unit)
        self.r = ((self.r * u.Unit(zunit)).to(u.meter)).value
        self.zunit = u.meter

        # Works with ellispoid reference so if 'EGM96' (geoid) we translate to 'WGS84' (ellipsoid) by adding EGM96
        # Let Oe be the ellipsoid WGS84 origin and Og be the EGM96 geoid one, then if a point M altitude is referred to
        # inside geoid reference, and we want to translate it to WGS84 reference then we write
        # OeM = OgM + OeOg with OeOg being the egm96-15 datum, OeM what we want and OgM what we have.
        if ref == 'EGM96':
            xP = np.arange(self.nx) -0.5
            yP = np.arange(self.nx) -0.5
            if self.srs.IsProjected():
                xGeo, yGeo = self.px_to_coord(xP, yP)
                lons, lats = self.proj(xGeo, yGeo, inverse=True)
            else:
                lons, lats = self.px_to_coord(xP, yP)
            egm96 = _A3DEGM96Manager()
            self.r += egm96(lons, lats)

        # create a in Memory dataset since we might have change self.r here, so it needs to be saved and linked with a dataset
        self.save_geotiff('')

    def get_slope_and_aspect(self, degree=False):
        """
        Return slope and aspect numpy arrays

        Slope & Aspects are computed as presented here :
        http://pro.arcgis.com/fr/pro-app/tool-reference/spatial-analyst/how-aspect-works.htm

        :param degree: boolean, set to True if returned slope in degree (False, default value, for percentage)
        :return: dem slope numpy array
        """

        # We need to define the scale factor between plani and alti resolution
        # -> we use the gdal spatial reference to know if our dem projected or not
        if self.srs.IsProjected() == 0:
            # Our dem is not projected, we can't simply use the pixel resolution
            # -> we need to compute resolution between each point
            lon, lat = self.px_to_coord()
            lonr = np.roll(lon,1,1)
            latl = np.roll(lat,1,0)

            distx = self._get_orthodromic_distance(lon, lat, lonr, lat)
            disty = self._get_orthodromic_distance(lon, lat, lon, latl)

            # deal withs ingularities at edges
            distx[:,0] = distx[:,1]
            disty[0] = disty[1]
        else:
            distx = np.abs(self.xres)
            disty = np.abs(self.yres)

        # Prepare convolving filters to compute the slope
        # Note that :
        #       * scipy.signal.convolve does not deal with np.nan
        #       * astropy.convolution.convolve does not deal with filter whose coefficients sum is 0 (which is our case)
        #       * numpy.convolve does not deal with 2D array so we would have to separate our slope filter
        #         and perform two numpy.convolve, that will do it but we eventually found out :
        #       * scipy.ndimages.filters.convolve ! And it looks like it does the job
        f1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) # for "gradient" x
        f2 = f1.transpose()                         # for "gradient" y

        # Now we do the convolutions :
        gx = filters.convolve(self.r, f1, mode='reflect')
        gy = filters.convolve(self.r, f2, mode='reflect')

        # And eventually we do compute tan(slope) and aspect
        tan_slope = np.sqrt((gx/distx)**2 + (gy/disty)**2)/8
        slope = np.arctan(tan_slope)
        aspect = np.arctan2(gy, gx)

        # Just simple unit change as required
        if degree is False:
            slope *= 100
        else :
            slope = (slope*180)/np.pi

        return slope, aspect


class _A3DEGM96Manager(object):
    """
    Deal with EGM96 Geoid superimposition
    """
    def __init__(self):
        egm_ds = gdal.Open('/work/logiciels/atelier3D/Data/egm/egm96_15.gtx')

        (x_start, x_step, dontcare0, y_start, dontcare1, y_step) = egm_ds.GetGeoTransform()
        self._nx = egm_ds.RasterXSize
        self._ny = egm_ds.RasterYSize

        # Get x_egm and y_egm ready for lon, -lat interpolation
        self._x_egm = x_start + x_step * np.arange(self._nx)
        self._y_egm = y_start + y_step * np.arange(self._ny)
        self._y_egm *= -1.0

        self._z_egm = egm_ds.ReadAsArray(0, 0, self._nx, self._ny)

    def __call__(self, x_new, y_new):
        """
        Return egm96 interpolated to x_new, y_new mesh

        :param x_new: lons
        :param y_new: lats
        :return: egm96(x_new, y_new)
        """

        # Set up coords
        y = -1.0 * y_new
        x = x_new.copy()
        x[x<0] += 360.0

        # link x,y to coords
        xr = (self._nx - 1) * (x - self._x_egm[0]) / (self._x_egm[-1] - self._x_egm[0])
        yr = (self._ny - 1) * (y - self._y_egm[0]) / (self._y_egm[-1] - self._y_egm[0])
        xr = np.clip(xr, 0, self._nx - 1)
        yr = np.clip(yr, 0, self._ny - 1)

        xi0 = xr.astype(np.int32)
        yi0 = yr.astype(np.int32)
        xi1 = np.clip(xi0+1,0,self._nx-1)
        yi1 = np.clip(yi0+1,0,self._ny-1)

        delta_x = xr - xi0.astype(np.float32)
        delta_y = yr - yi0.astype(np.float32)

        z_new = self._z_egm[yi0, xi0] * (1.-delta_x) * (1.-delta_y) + \
                self._z_egm[yi1, xi1] * delta_x * delta_y + \
                self._z_egm[yi1, xi0] * (1.-delta_x) * delta_y + \
                self._z_egm[yi0, xi1] * delta_x * (1.-delta_y)

        return z_new


def write_geotiff(outputFile, raster, geoTransform, wkt=None, proj4=None, mask=None, dtype=gdal.GDT_Float32, nodata=-999):
    """

    :param outfile:
    :param raster: numpy array
    :param geoTransform:
    :param wkt: a WKT projection string
    :param proj4: a proj4 string
    :param mask: mask band
    :param dtype: GDAL type
    :param nodata: nodata value to set
    :return:

    Based on http://adventuresindevelopment.blogspot.com/2008/12/python-gdal-adding-geotiff-meta-data.html
    and http://www.gdal.org/gdal_tutorial.html and https://github.com/atedstone/georaster
    """

    # Georeferencing sanity checks
    if wkt and proj4:
        raise NameError('ERROR: Both wkt and proj4 specified. Only specify one.')
    if wkt is None and proj4 is None:
        raise NameError('ERROR: One of wkt or proj4 need to be specified.')

    # Check if the image is multi-band or not.
    if raster.shape.__len__() == 2:
        nbands = 1
        ydim = raster.shape[0]
        xdim = raster.shape[1]
    else:
        raise NameError('ERROR: Raster shape can only be two (y, x)')

    # Setup geotiff file.
    if outputFile != '':
        driver = gdal.GetDriverByName("GTiff")
    else:
        driver = gdal.GetDriverByName('MEM')

    dst_ds = driver.Create(outputFile, xdim, ydim, nbands, dtype)
    # Top left x, w-e pixel res, rotation, top left y, rotation, n-s pixel res
    dst_ds.SetGeoTransform(geoTransform)

    # Set the reference info
    srs = osr.SpatialReference()
    if wkt:
        dst_ds.SetProjection(wkt)
    elif proj4:
        srs.ImportFromProj4(proj4)
        dst_ds.SetProjection(srs.ExportToWkt())

    # Write array
    dst_ds.GetRasterBand(1).WriteArray(raster)
    dst_ds.GetRasterBand(1).SetNoDataValue(nodata)
    if mask:
        dst_ds.GetRasterBand(1).GetMaskBand().WriteArray(mask)

    if outputFile != '':
        dst_ds = None
        return True
    else:
        return dst_ds
