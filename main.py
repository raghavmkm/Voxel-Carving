
import scipy.io
import numpy as np
import cv2
import glob
import vtk
import functions
import matplotlib.pyplot as plt
plt.rc('figure', max_open_warning = 0)


# main function
def main():
    projections, images, height, width = functions.read_input()
    silhouettes = functions.silhouette_images(images)
    grid = functions.init_grid()
    occupancy, voxels = functions.carve(projections, silhouettes, grid, height, width)
    #%% save as rectilinear grid (this enables paraview to display its iso-volume as a mesh)
    x = voxels[::120*120, 0]
    y = voxels[:120*120:120, 1]
    z = voxels[:120, 2]

    xCoords = vtk.vtkFloatArray()
    for i in x:
        xCoords.InsertNextValue(i)
    yCoords = vtk.vtkFloatArray()
    for i in y:
        yCoords.InsertNextValue(i)
    zCoords = vtk.vtkFloatArray()
    for i in z:
        zCoords.InsertNextValue(i)

    values = vtk.vtkFloatArray()
    for i in occupancy:
        values.InsertNextValue(i)
        
    rgrid = vtk.vtkRectilinearGrid()
    rgrid.SetDimensions(len(x), len(y), len(z))
    rgrid.SetXCoordinates(xCoords)
    rgrid.SetYCoordinates(yCoords)
    rgrid.SetZCoordinates(zCoords)
    rgrid.GetPointData().SetScalars(values)

    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName("output.vtr")
    writer.SetInputData(rgrid)
    writer.Write()

if __name__=="__main__":
    main()