import numpy as np

def vvg_df_to_centerline_array(vvg_df_edges,vvg_df_nodes, shape):

    cl_arr = np.zeros(shape, dtype=np.uint8)
    for cl in vvg_df_edges["pos"]:
        for pos in cl:
            cl_arr[int(pos[0]), int(pos[1])] = 1

    for i, cl in enumerate(vvg_df_nodes["voxel_pos"]):
        for pos in cl:
            cl_arr[int(pos[0]), int(pos[1])] = 1

            if vvg_df_nodes["radius"][i] >1:
                cl_arr = color_neighbor_pixels(cl_arr, pos, vvg_df_nodes["radius"][i])

    return cl_arr




"""def color_neighbor_pixels(arr, pos, radius):
    # set all pixels in arr around pos to 1
    # radius is the radius of the circle
    # pos is the center of the circle
    # arr is the array to be modified

    radius = int(radius)
    print(radius)
    # get the coordinates of the circle
    x, y = np.ogrid[-radius: radius + 1, -radius: radius + 1]

    # get the mask of the circle
    mask = x ** 2 + y ** 2 <= radius ** 2

    # set the pixels in the mask to 1
    arr[int(pos[0]) - radius: int(pos[0]) + radius + 1, int(pos[1]) - radius: int(pos[1]) + radius + 1][mask] = 1

    return arr"""

def color_neighbor_pixels(arr, pos, radius):
    # set all pixels in arr around pos to 1
    # radius is the radius of the circle
    # pos is the center of the circle
    # arr is the array to be modified
    radius = int(radius/1.75)+1

    if radius <1:
        radius = 1

    #print(radius)

    # Get the coordinates of the circle
    x, y = np.ogrid[-radius: radius + 1, -radius: radius + 1]

    # Get the mask of the circle, clipped to match the slice size
    mask = x ** 2 + y ** 2 <= radius ** 2

    # Set the pixels in the masked area to 1
    x_start = max(int(pos[0]) - radius, 0)
    x_end = min(int(pos[0]) + radius + 1, arr.shape[0])
    y_start = max(int(pos[1]) - radius, 0)
    y_end = min(int(pos[1]) + radius + 1, arr.shape[1])

    x_range = x_end - x_start
    y_range = y_end - y_start

    #check if the start or end are outside the array
    if x_start == 0 and int(pos[0]) - radius != 0:
        x_lower_bound = True
    else:
        x_lower_bound = False
    
    if y_start == 0 and int(pos[1]) - radius != 0:
        y_lower_bound = True
    else:
        y_lower_bound = False
    
    if x_end == arr.shape[0] and int(pos[0]) + radius + 1 != arr.shape[0]:
        x_upper_bound = True
    else:
        x_upper_bound = False

    if y_end == arr.shape[1] and int(pos[1]) + radius + 1 != arr.shape[1]:
        y_upper_bound = True
    else:
        y_upper_bound = False

    if x_lower_bound:
        mask = mask[-x_range:,]
    if y_lower_bound:
        mask = mask[:,-y_range:]

    if x_upper_bound:
        mask = mask[:x_range,]
    if y_upper_bound:
        mask = mask[:,:y_range]


    arr_slice = arr[x_start:x_end, y_start:y_end]
    #mask = mask[:arr_slice.shape[0], :arr_slice.shape[1]]
    arr[x_start:x_end, y_start:y_end][mask] = 1

    return arr