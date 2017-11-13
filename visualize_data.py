import cv2
import numpy as np
import model
import matplotlib.pyplot as plt

plt.axis('off')

image = cv2.imread("/Users/iraadit/Datasets/Udacity/Behavorial Cloning/data/IMG/center_2016_12_01_13_33_46_143.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
#plt.show()

image_brightness = model.random_brightness(image)
plt.imshow(image_brightness)
plt.savefig("bright.png")
#plt.show()

image_shadow = model.random_shadow(image)
plt.imshow(image_shadow)
plt.savefig("shadow.png")
#plt.show()
