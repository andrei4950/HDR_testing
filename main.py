from PIL import Image
import numpy as np


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
    return hi_res_1, hi_res_2, hi_res_3, low_res_1, low_res_2, low_res_3


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


if __name__ == '__main__':
    pass
