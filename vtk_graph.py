# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2

from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints


from vtkmodules.vtkCommonDataModel import vtkMutableUndirectedGraph, vtkPolyData
from vtkmodules.vtkFiltersSources import vtkGraphToPolyData
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
import pyvista





colors = vtkNamedColors()
# Create a graph
g = vtkMutableUndirectedGraph()
# Add 4 vertices to the graph
v1 = g.AddVertex()
v2 = g.AddVertex()
v3 = g.AddVertex()
v4 = g.AddVertex()
# Add 3 edges to the graph
g.AddEdge(v1, v2)
g.AddEdge(v1, v3)
g.AddEdge(v1, v4)
# Create 4 points - one for each vertex

points = vtkPoints()
points.InsertNextPoint(1.0, 0.0, 0.0)
points.InsertNextPoint(1.0, 0.0, 0.0)
points.InsertNextPoint(0.0, 1.0, 0.0)
points.InsertNextPoint(0.0, 1.0, 1.0)
# Add the coordinates of the points to the graph
g.SetPoints(points)
# Convert the graph to a polydata
graphToPolyData = vtkGraphToPolyData()
graphToPolyData.SetInputData(g)
graphToPolyData.Update()
# Create a mapper and actor
mapper = vtkPolyDataMapper()
mapper.SetInputConnection(graphToPolyData.GetOutputPort())

actor = vtkActor()
actor.SetMapper(mapper)


actor.GetProperty().SetColor(1, 0, 0)
actor.GetProperty().SetEdgeVisibility(1)
actor.GetProperty().SetEdgeColor(0.9,0.9,0.4)
actor.GetProperty().SetLineWidth(6)
actor.GetProperty().SetPointSize(12)
actor.GetProperty().SetRenderLinesAsTubes(1)
actor.GetProperty().SetRenderPointsAsSpheres(1)
actor.GetProperty().SetVertexVisibility(1)
actor.GetProperty().SetVertexColor(0.5,1.0,0.8)


p = pyvista.Plotter()
p.add_actor(actor)

p.show()


# Create a renderer, render window, and interactor
#renderer = vtkRenderer()
#renderWindow = vtkRenderWindow()
#renderWindow.AddRenderer(renderer)
#renderWindowInteractor = vtkRenderWindowInteractor()
#renderWindowInteractor.SetRenderWindow(renderWindow)
#
## Add the actor to the scene
#renderer.AddActor(actor)
#renderer.SetBackground(colors.GetColor3d('White'))
#
## Render and interact
#renderWindow.SetWindowName('GraphToPolyData')
#renderWindow.Render()
#renderWindowInteractor.Start()

#print(type(graphToPolyData.GetOutputPort()))

#print(dir(graphToPolyData))
#print(type(graphToPolyData.GetOutput()))
#pyPoly = pyvista.PolyData(graphToPolyData.GetOutput())

#print(pyPoly)


#p = pyvista.Plotter()

#p.add_mesh(pyPoly, style='points',  render_lines_as_tubes=False, render_points_as_spheres=True, point_size=100, line_width=10)
#p.add_mesh(pyPoly, style='surface',  render_lines_as_tubes=True, render_points_as_spheres=False, point_size=100, line_width=10)

#p.show()
