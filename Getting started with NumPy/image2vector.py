import numpy as np

# in computer science, an image is represented by a 3D array of shape  (length,height,depth=3)(length,height,depth=3) .
#  However, when you read an image as the input of an algorithm you convert it to a vector of shape  (length∗height∗3,1)

def image2vector(image):

    v=image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)

    return v

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print(image2vector(image))

