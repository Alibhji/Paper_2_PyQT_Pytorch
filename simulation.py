import numpy as np
import random

def generate_random_data(height, width, count):
    x, y = zip(*[generate_img_and_mask(height, width) for i in range(0, count)])

    X = np.asarray(x) * 255
    X = X.repeat(3, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
    Y = np.asarray(y)

    return X, Y

def generate_img_and_mask(height, width,triangle=False,circle=True,mesh=False,square=False,plus=False):
    shape = (height, width)
    # Create input image
    arr = np.zeros(shape, dtype=bool)
    mask_list=[]

    if(triangle):
        triangle_location = get_random_location(*shape)
        arr = add_triangle(arr, *triangle_location)
        mask_list.append(add_triangle(np.zeros(shape, dtype=bool), *triangle_location))

    if (circle):
        circle_location1 = get_random_location(*shape, zoom=0.7)
        arr = add_circle(arr, *circle_location1)
        mask_list.append(add_circle(np.zeros(shape, dtype=bool), *circle_location1))
        # arr = add_circle(arr, *circle_location2, fill=True)
        # circle_location2 = get_random_location(*shape, zoom=0.5)
        # mask_list.append(add_circle(np.zeros(shape, dtype=bool), *circle_location2, fill=True))


    if (mesh):
        mesh_location = get_random_location(*shape)
        arr = add_mesh_square(arr, *mesh_location)
        mask_list.append(add_filled_square(np.zeros(shape, dtype=bool), *mesh_location))

    if (square):
        square_location = get_random_location(*shape, zoom=0.8)
        arr = add_filled_square(arr, *square_location)
        mask_list.append(add_filled_square(np.zeros(shape, dtype=bool), *square_location))

    if (plus):
        plus_location = get_random_location(*shape, zoom=1.2)
        arr = add_plus(arr, *plus_location)
        mask_list.append(add_plus(np.zeros(shape, dtype=bool), *plus_location))


    arr = np.reshape(arr, (1, height, width)).astype(np.float32)

    # Create target masks
    masks = np.asarray(mask_list).astype(np.float32)

    return arr, masks

def add_square(arr, x, y, size):
    s = int(size / 2)
    arr[x-s,y-s:y+s] = True
    arr[x+s,y-s:y+s] = True
    arr[x-s:x+s,y-s] = True
    arr[x-s:x+s,y+s] = True

    return arr

def add_filled_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s]))

def logical_and(arrays):
    new_array = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        new_array = np.logical_and(new_array, a)

    return new_array

def add_mesh_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]))

def add_triangle(arr, x, y, size):
    s = int(size / 2)

    triangle = np.tril(np.ones((size, size), dtype=bool))

    arr[x-s:x-s+triangle.shape[0],y-s:y-s+triangle.shape[1]] = triangle

    return arr

def add_circle(arr, x, y, size, fill=False):
    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
    circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))

    return new_arr

def add_plus(arr, x, y, size):
    s = int(size / 2)
    arr[x-1:x+1,y-s:y+s] = True
    arr[x-s:x+s,y-1:y+1] = True

    return arr

def get_random_location(width, height, zoom=1.0):
    x = int(width * random.uniform(0.1, 0.9))
    y = int(height * random.uniform(0.1, 0.9))

    size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

    return (x, y, size)