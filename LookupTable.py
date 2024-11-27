import numpy as np
import cv2

def create_lut(h,s,l,size=256):
    h_range=2
    s_range=2
    l_lange=20



    lut = np.zeros((size, size,3), dtype=np.uint8)

    transformed_lr = l-l_lange




    for illumination in range(size):

        transformed_h = h - h_range
        transformed_s = s - s_range
        transformed_l = transformed_lr - l_lange


        transformed_lr = transformed_lr + (l_lange / 150) * 2

        transformed_lr = np.clip(transformed_lr, 0, l)

        for saturation in range(size):


            transformed_h=transformed_h+(h_range/100)
            transformed_h=np.clip(transformed_h, 0, h)
            transformed_s=transformed_s+(s_range/200)
            transformed_l=transformed_l+(l_lange/150)*2

            transformed_l = np.clip(transformed_l, 0, l)


            # Store transformed value in the LUT
            lut[saturation,illumination ] = [transformed_h,transformed_l,s]

    return lut





# Example usage
h=241
s=67*255/100
original_l=52 *255/100 -10



def map_range(value, in_min, in_max, out_min, out_max):
    # Map the input value from the input range to the output range
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# Example usage
input_value = 90  # Input value between 0 and 180
h = map_range(h, 0, 360, 0, 180)
print("Mapped value:", h)
l=original_l
lower_l=l
upper_l=l
#l=l*255/100
if l<50:
    l=l+30
    upper_l=l

else:
    l=l-30
    lower_l=l



def makegrid(h,l,s):
    new_h = h
    new_l = l

    if new_h < 0:
        new_h = 0
    if new_l < 0:
        new_l = 0

    grid_size = 8
    tile_size = 64
    larger_image_width = tile_size * grid_size
    larger_image_height = tile_size * grid_size

    # Create a blank larger image
    larger_image = np.zeros((larger_image_height, larger_image_width, 3), dtype=np.uint8)

    for i in range(0, grid_size):

        lut = create_lut(new_h, s, new_l, tile_size)
        lut = cv2.cvtColor(lut, cv2.COLOR_HLS2BGR)

        row = i % grid_size

        for j in range(0, grid_size):
            col = j % grid_size
            start_y = row * tile_size
            end_y = start_y + tile_size
            start_x = col * tile_size
            end_x = start_x + tile_size

            larger_image[start_y:end_y, start_x:end_x, :] = lut

        print("Processing image:" + str(i))
        new_h = new_h + 1
        new_l = new_l + 5

        new_h = np.clip(new_h, 0, h)
        new_l = np.clip(new_l, 0, l)
    return larger_image




from PIL import Image


def average_luminance(image_path):
    # Open the image
    img = Image.open(image_path)

    # Convert the image to grayscale
    img_gray = img.convert('L')

    # Calculate the average luminance
    luminance_values = list(img_gray.getdata())
    average_luminance = sum(luminance_values) / len(luminance_values)

    return average_luminance


def map_luminance(average_intensity, lower_luminance, upper_luminance):
    upper_intensity=255
    lower_intensity=0
    # Calculate the total intensity range
    intensity_range = upper_intensity - lower_intensity

    # Calculate the luminance range
    luminance_range = upper_luminance - lower_luminance

    # Calculate the scaled intensity within the range [0, 1]
    scaled_intensity = (average_intensity - lower_intensity) / intensity_range

    # Map the scaled intensity to the luminance range
    mapped_luminance = lower_luminance + (scaled_intensity * luminance_range)

    return mapped_luminance

luminance_range=30
# Example usage
image_path = '/home/gulraiz/Downloads/Golden-brunette-balayage-13.jpg'  # Replace with the path to your image
avg_luminance = average_luminance(image_path)
print("Average Luminance:", avg_luminance)
print("Actual Luminance:"+str(original_l))
print("Lower Luminance:"+str(lower_l))
print("Upper Luminance:"+str(upper_l))
new_luminance=map_luminance(avg_luminance,lower_l, upper_l)
print("Change luminance:",new_luminance )

larger_image_lower=makegrid(h,lower_l,s)
larger_image_upper=makegrid(h,upper_l,s)

larger_image=makegrid(h,new_luminance,s)

cv2.imshow('LUT Image Dark', larger_image_lower)

cv2.imshow('LUT Image Light', larger_image_upper)
cv2.imshow('Applied Image', larger_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Golden-brunette-balayage-13

