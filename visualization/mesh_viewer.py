import numpy as np 
import pyvista


hexcolors = {
    'aliceblue': '#F0F8FF',    'antiquewhite': '#FAEBD7','aquamarine': '#7FFFD4',    'azure': '#F0FFFF',    'beige': '#F5F5DC',    'bisque': '#FFE4C4','black': '#000000',    'blanchedalmond': '#FFEBCD',    'blue': '#0000FF',    'blueviolet': '#8A2BE2',    'brown': '#654321',    'burlywood': '#DEB887',    'cadetblue': '#5F9EA0',    'chartreuse': '#7FFF00',    'chocolate': '#D2691E',    'coral': '#FF7F50',
    'cornflowerblue': '#6495ED',    'cornsilk': '#FFF8DC',    'crimson': '#DC143C',    'cyan': '#00FFFF',    'darkblue': '#00008B',    'darkcyan': '#008B8B',    'darkgoldenrod': '#B8860B',    'darkgray': '#A9A9A9',    'darkgreen': '#006400',    'darkkhaki': '#BDB76B',    'darkmagenta': '#8B008B',   'darkolivegreen': '#556B2F',    'darkorange': '#FF8C00',    'darkorchid': '#9932CC',    'darkred': '#8B0000',    'darksalmon': '#E9967A',    'darkseagreen': '#8FBC8F',
    'darkslateblue': '#483D8B',    'darkslategray': '#2F4F4F',    'darkturquoise': '#00CED1',    'darkviolet': '#9400D3',
    'deeppink': '#FF1493',    'deepskyblue': '#00BFFF',
    'dimgray': '#696969',    'dodgerblue': '#1E90FF',
    'firebrick': '#B22222',
    'floralwhite': '#FFFAF0',
    'forestgreen': '#228B22',
    'gainsboro': '#DCDCDC',
    'ghostwhite': '#F8F8FF',
    'gold': '#FFD700',
    'goldenrod': '#DAA520',
    'gray': '#808080',
    'green': '#008000',
    'greenyellow': '#ADFF2F',
    'honeydew': '#F0FFF0',
    'hotpink': '#FF69B4',
    'indianred': '#CD5C5C',
    'indigo': '#4B0082',
    'ivory': '#FFFFF0',
    'khaki': '#F0E68C',
    'lavender': '#E6E6FA',
    'lavenderblush': '#FFF0F5',
    'lawngreen': '#7CFC00',
    'lemonchiffon': '#FFFACD',
    'lime': '#00FF00',
    'limegreen': '#32CD32',
    'linen': '#FAF0E6',
    'magenta': '#FF00FF',
    'maroon': '#800000',
    'mediumaquamarine': '#66CDAA',
    'mediumblue': '#0000CD',
    'mediumorchid': '#BA55D3',
    'mediumpurple': '#9370DB',
    'mediumseagreen': '#3CB371',
    'mediumslateblue': '#7B68EE',
    'mediumspringgreen': '#00FA9A',
    'mediumturquoise': '#48D1CC',
    'mediumvioletred': '#C71585',
    'midnightblue': '#191970',
    'mintcream': '#F5FFFA',
    'mistyrose': '#FFE4E1',
    'moccasin': '#FFE4B5',
    'navajowhite': '#FFDEAD',
    'navy': '#000080',
    'oldlace': '#FDF5E6',
    'olive': '#808000',
    'olivedrab': '#6B8E23',
    'orange': '#FFA500',
    'orangered': '#FF4500',
    'orchid': '#DA70D6',
    'palegoldenrod': '#EEE8AA',
    'palegreen': '#98FB98',
    'paleturquoise': '#AFEEEE',
    'palevioletred': '#DB7093',
    'papayawhip': '#FFEFD5',
    'paraview_background': '#52576e',
    'peachpuff': '#FFDAB9',
    'peru': '#CD853F',
    'pink': '#FFC0CB',
    'plum': '#DDA0DD',
    'powderblue': '#B0E0E6',
    'purple': '#800080',
    'raw_sienna': '#965434',
    'rebeccapurple': '#663399',
    'red': '#FF0000',
    'rosybrown': '#BC8F8F',
    'royalblue': '#4169E1',
    'saddlebrown': '#8B4513',
    'salmon': '#FA8072',
    'sandybrown': '#F4A460',
    'seagreen': '#2E8B57',
    'seashell': '#FFF5EE',
    'sienna': '#A0522D',
    'silver': '#C0C0C0',
    'skyblue': '#87CEEB',
    'slateblue': '#6A5ACD',
    'slategray': '#708090',
    'snow': '#FFFAFA',
    'springgreen': '#00FF7F',
    'steelblue': '#4682B4',
    'tan': '#D2B48C',
    'teal': '#008080',
    'thistle': '#D8BFD8',
    'tomato': '#FF6347',
    'turquoise': '#40E0D0',
    'violet': '#EE82EE',
    'wheat': '#F5DEB3',
    'white': '#FFFFFF',
    'whitesmoke': '#F5F5F5',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32',
    'tab:blue': '#1f77b4',
    'tab:orange': '#ff7f0e',
    'tab:green': '#2ca02c',
    'tab:red': '#d62728',
    'tab:purple': '#9467bd',
    'tab:brown': '#8c564b',
    'tab:pink': '#e377c2',
    'tab:gray': '#7f7f7f',
    'tab:olive': '#bcbd22',
    'tab:cyan': '#17becf',
}




def show2_trimeshs(trimesh1, trimesh2):
    """" Creates an interactive visualization of two trimesh objects. 
    :trimesh1/2: Two trimesh objects
    
    """

    mesh1 = pyvista.wrap(trimesh1)
    mesh2 = pyvista.wrap(trimesh2)

    plotter = pyvista.Plotter()

    actor = plotter.add_mesh(mesh1, color='red')
    actor = plotter.add_mesh(mesh2, color='yellow')

    print("File 1 is red")
    print("File 2 is yellow")

    plotter.show()



def show3_trimeshs(trimesh1, trimesh2, trimesh3):
    """" Creates an interactive visualization of two trimesh objects. 
    trimesh1/2/3: Three trimesh objects
    
    """

    mesh1 = pyvista.wrap(trimesh1)
    mesh2 = pyvista.wrap(trimesh2)
    mesh3 = pyvista.wrap(trimesh3)

    plotter = pyvista.Plotter()

    actor = plotter.add_mesh(mesh1, color='red')
    actor = plotter.add_mesh(mesh2, color='yellow')
    actor = plotter.add_mesh(mesh3, color='green')

    print("File 1 is red")
    print("File 2 is yellow")
    print("File 3 is green")

    plotter.show()



def showList_trimesh(trimesh_lis, color_list = None):

    if color_list is None:
        color_list = np.random.choice(list(hexcolors.keys()), size=len(trimesh_lis), replace=True, p=None)
    elif len(color_list) < len(trimesh_lis):
        raise ValueError("More colors than meshes must be provided!")

    plotter = pyvista.Plotter()
    for i in range(len(trimesh_lis)):
        mesh = pyvista.wrap(trimesh_lis[i])
        actor = plotter.add_mesh(mesh, color = color_list[i])
    plotter.show()



def gifList_trimesh(trimesh_lis, gif_name):
    col_sel = np.random.choice(list(hexcolors.keys()), size=len(trimesh_lis), replace=True, p=None)
    plotter = pyvista.Plotter()
    for i in range(len(trimesh_lis)):
        mesh = pyvista.wrap(trimesh_lis[i])
        plotter.add_mesh(mesh, color=col_sel[i])

    viewup = [1, 1, 1]
    path = plotter.generate_orbital_path(factor=2.0, n_points=72, viewup=viewup, shift=0.1)
    plotter.open_gif(gif_name + ".gif")
    plotter.orbit_on_path(path, write_frames=True, viewup=viewup, step=0.05)
    plotter.close()

