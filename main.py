import numpy
from PIL import Image
import numpy as np
from matplotlib import pyplot
from scipy.optimize import curve_fit


def is_overexposed(pixel):
    for i in range(3):
        if pixel[i] > 220:
            return True
    return False


def process(numpydata):
    height = numpydata.shape[0]
    width = numpydata.shape[1]
    out = np.zeros((height, width, 3)).astype('uint8')
    tot_count = width * height
    count = 0
    for line in range(height):
        print("processing {:.0f} %".format(count / tot_count * 100), end='\r', flush=True)
        for column in range(width):
            count += 1
            if is_overexposed(numpydata[line][column]):
                out[line][column] = [255, 0, 0]
            else:
                out[line][column] = numpydata[line][column]
    return out


# load the image and convert into
# numpy array
def load_test_images():
    hi_res_1 = Image.open('testing resources/hi_res_1.JPG')
    hi_res_2 = Image.open('testing resources/hi_res_2.JPG')
    hi_res_3 = Image.open('testing resources/hi_res_3.JPG')
    low_res_1 = Image.open('testing resources/low_res_1.JPG')
    low_res_2 = Image.open('testing resources/low_res_2.JPG')
    low_res_3 = Image.open('testing resources/low_res_3.JPG')
    test_b = Image.open('testing resources/test_bright.jpeg')
    test_m = Image.open('testing resources/test_med.jpeg')
    test_d = Image.open('testing resources/test_dark.jpeg')

    return hi_res_1, hi_res_2, hi_res_3, low_res_1, low_res_2, low_res_3, test_b, test_m, test_d


def scale_brightness(img, a):
    height = img.shape[0]
    width = img.shape[1]
    out = np.zeros((height, width, 3)).astype('uint8')
    tot_count = width * height
    count = 0
    for line in range(height):
        print("scaling {:.0f} %".format(count / tot_count * 100), end='\r', flush=True)
        for column in range(width):
            count += 1
            for colour in range(3):
                out[line][column][colour] = min(255, img[line][column][colour] * a)
    return out

"""
test_images = load_test_images()
numpydata = np.asarray(test_images[4])

out = scale_brightness(numpydata, 1.8)

print(type(numpydata[0][0][0]))
print(numpydata.shape)
print(type(out[0][0][0]))
print(out.shape)

image = Image.fromarray(out, 'RGB')
# Save the image
image.save('image.png')
"""


def quadratic(x, a, b, c):
    return np.clip(a*x**2+b*x+c, 0, 255)


def sqrt(x, a, b, c):
    return np.clip(a*x+b*x**0.5+c, 0, 255)


def cubic_compact(x, params, clipping = True):
    return cubic(x, params[0], params[1], params[2], params[3], clipping=clipping)


def cubic(x, a, b, c, d, clipping = True):
    out = a*x**3+b*x**2+c*x+d
    if clipping:
        return np.clip(out, 0, 255)
    else:
        return out


def get_weights(img: np.array):
    one = np.ones(img.shape)
    return one*20 - (img - one*128)**2/850


def make_hdr(img_array):
    # get shape and make sure images are all the same shape
    shape = None
    for img in img_array:
        if shape is None:
            shape = img.shape
        else:
            if shape != img.shape:
                raise Exception("Images have different shapes {}, {}".format(shape, img.shape))

    # flatten images
    for i in range(len(img_array)):
        img_array[i] = img_array[i].flatten().astype('float64')
    print("images: ")
    print(img_array)

    # sort images

    # obtain scalings
    relative_scalings = []
    for i in range(1, len(img_array)):
        params, covariant_matrix = curve_fit(f=cubic, xdata=img_array[i-1], ydata=img_array[i], p0=[0, 0, 2, 10])
        relative_scalings.append(params)
    print("scalings: ")
    print(relative_scalings)

    # get weights
    weights = []
    for img in img_array:
        weights.append(get_weights(img))
    print("weights: ")
    print(weights)

    # scale up all images
    for img_i in range(len(img_array)-1):
        for scaling_i in range(img_i, len(img_array)-1):
            img_array[img_i] = cubic_compact(img_array[img_i], relative_scalings[scaling_i], clipping=False)
    print("scaled images: ")
    print(img_array)

    # average images
    sum = numpy.zeros(shape).flatten()
    tot_weights = numpy.zeros(shape).flatten()
    for i in range(len(img_array)):
        sum += img_array[i] * weights[i]
        tot_weights += weights[i]
    print("sum: ")
    print(sum)
    hdr = sum / tot_weights
    print("hdr: ")
    print(hdr)
    print(hdr.max())
    hdr = hdr / hdr.max() * 255

    return Image.fromarray(hdr.reshape(shape).astype('uint8'), 'RGB')


if __name__ == '__main__':
    bright_hd, med_hd, dark_hd, bright, med, dark, t_b, t_m, t_d = load_test_images()

    hdr = make_hdr([np.asarray(t_d), np.asarray(t_b)])
    hdr.save('image.png')

    """np_darker = np.asarray(t_d)
    np_brighter = np.asarray(t_m)
    shape = np_darker.shape
    print(shape)
    x = np_darker.flatten()
    y = np_brighter.flatten()

    pyplot.scatter(x, y, s=1)

    print("starting fit")
    params, covariant_matrix = curve_fit(f=quadratic, xdata=x, ydata=y, p0=[-0.004, 1.97, 10])
    parameter_uncertainties = np.sqrt(covariant_matrix.diagonal())
    print(params)
    print(parameter_uncertainties)
    pyplot.plot(np.arange(255), quadratic(np.arange(255), params[0], params[1], params[2]), color='red')

    print("starting fit 2")
    params, covariant_matrix = curve_fit(f=sqrt, xdata=x, ydata=y, p0=[0, 23, 19])
    parameter_uncertainties = np.sqrt(covariant_matrix.diagonal())
    print(params)
    print(parameter_uncertainties)
    pyplot.plot(np.arange(255), sqrt(np.arange(255), params[0], params[1], params[2]), color='pink')

    print("starting fit 3")
    params, covariant_matrix = curve_fit(f=cubic, xdata=x, ydata=y, p0=[0, 0, 2, 10])
    parameter_uncertainties = np.sqrt(covariant_matrix.diagonal())
    print(params)
    print(parameter_uncertainties)
    pyplot.plot(np.arange(255), cubic(np.arange(255), params[0], params[1], params[2], params[3]), color='orange')

    pyplot.show()

    out = cubic(x, params[0], params[1], params[2], params[3])
    out = np.ones(out.shape)*128 + out - y
    image = Image.fromarray(out.reshape(shape).astype('uint8'), 'RGB')
    image.save('image.png')"""

