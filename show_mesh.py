import pandas as pd
import preprocessing as pp
from visualization import graph_to_mesh, mesh_viewer



nodesFile =  "~/Documents/Mcao3_new/nodes_mcao3_fill_ones_3.csv"
edgesFile = "~/Documents/Mcao3_new/edges_mcao3_fill_ones_3.csv"



nodes = pd.read_csv(nodesFile, sep = ";", index_col= "id")
edges = pd.read_csv(edgesFile, sep = ";", index_col= "id")

nodes = pp.scalePosition(nodes, (1.65,1.65,6))


mesh_viewer.renderGraph(nodes, edges, vtk = 0)




#import vtkmodules.vtkInteractionStyle
#import vtkmodules.vtkRenderingOpenGL2
#from vtkmodules.vtkCommonColor import vtkNamedColors
#from vtkmodules.vtkCommonCore import vtkPoints
#from vtkmodules.vtkCommonCore import vtkLookupTable
#from vtkmodules.vtkCommonDataModel import vtkMutableUndirectedGraph
#from vtkmodules.vtkFiltersSources import vtkGraphToPolyData
##from vtkmodules.
#
#from vtkmodules.vtkRenderingCore import (
#    vtkActor,
#    vtkPolyDataMapper,
#    vtkRenderWindow,
#    vtkRenderWindowInteractor,
#    vtkRenderer
#)
#
#colors = vtkNamedColors()
## Create a graph
#g = vtkMutableUndirectedGraph()
## Add 4 vertices to the graph
#v1 = g.AddVertex()
#v2 = g.AddVertex()
#v3 = g.AddVertex()
#v4 = g.AddVertex()
## Add 3 edges to the graph
#g.AddEdge(v1, v2)
#g.AddEdge(v1, v3)
#g.AddEdge(v1, v4)
## Create 4 points - one for each vertex
#points = vtkPoints()
#points.InsertNextPoint(0.0, 0.0, 0.0)
#points.InsertNextPoint(1.0, 0.0, 0.0)
#points.InsertNextPoint(0.0, 1.0, 0.0)
#points.InsertNextPoint(0.0, 0.0, 1.0)
## Add the coordinates of the points to the graph
#g.SetPoints(points)
## Convert the graph to a polydata
#graphToPolyData = vtkGraphToPolyData()
#graphToPolyData.SetInputData(g)
#graphToPolyData.Update()
## Create a mapper and actor
#
#table = vtkLookupTable()
#table.SetNumberOfTableValues(1)
#table.SetTableValue(0, 1, 0, 0)
#
#
#mapper = vtkPolyDataMapper()
#mapper.SetInputConnection(graphToPolyData.GetOutputPort())
#
#
##print(mapper.GetLookupTable().SetNumberOfTableValues(1))
##print(mapper.GetLookupTable().SetTableValue(0, 1, 0, 0))
##mapper.ScalarVisibilityOn()
##print(mapper.SetColorMode(1))
##print(mapper.GetColorMode())
#
#actor = vtkActor()
#actor.SetMapper(mapper)
#
#
#actor.GetProperty().SetRenderLinesAsTubes(1)
#actor.GetProperty().SetRenderPointsAsSpheres(1)
#actor.GetProperty().SetEdgeVisibility(1)
#actor.GetProperty().SetVertexVisibility(1)
#actor.GetProperty().SetLineWidth(2)
#actor.GetProperty().SetPointSize(5)
#actor.GetProperty().SetVertexColor(1,0,1)
#actor.GetProperty().SetEdgeColor(1,0,1)
#
#
## Create a renderer, render window, and interactor
#renderer = vtkRenderer()
#renderWindow = vtkRenderWindow()
#renderWindow.AddRenderer(renderer)
#renderWindowInteractor = vtkRenderWindowInteractor()
#renderWindowInteractor.SetRenderWindow(renderWindow)
## Add the actor to the scene
#renderer.AddActor(actor)
#renderer.SetBackground(colors.GetColor3d('Green'))
## Render and interact
#renderWindow.SetWindowName('GraphToPolyData')
#renderWindow.Render()
#renderWindowInteractor.Start()