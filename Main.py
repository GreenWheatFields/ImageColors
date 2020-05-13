import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import numpy as np
import time


# input image and run kmeans
def convert_image(file_path, clusters):
    global clt
    # gets image and turns it rbg

    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, depth = image.shape
    print(height)
    print(width)
    print(image.shape)
    # image = cv2.resize(image, (500, 500))
    height, width, depth = image.shape
    print(height)
    print(width)
    print(image.shape)

    # sets og image
    plt.figure()
    plt.axis("on")
    plt.imshow(image)

    # plt.show()

    # turn image to arrary of rgb value
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    start = time.time()  # time kmeans process

    print("working")
    clt = KMeans(clusters)  # define model
    clt.fit(image)  # train model
    print("done")
    end = time.time()
    print(end - start, end=" ")  # end timer

    print("seconds")
    return clt


def centroid_histogram(clt):
    # get the number of labels
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    # create histogram
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # sort and reverse histogram
    hist = np.sort(hist[::-1])
    hist = hist[::-1]
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster

    for (percent, color) in zip(hist, centroids):
        # define range based on percent dominance of a color
        endX = startX + (percent * 300)
        # create a rectangle, define the size, covert clusters to rgb, add to barchart
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        # define new beginning for next bar
        startX = endX

    return bar


def display_image(input_path, output_path, clusters):
    convert_image(input_path, clusters)
    # generate histogram,
    hist = centroid_histogram(clt)
    # generate barchart
    bar = plot_colors(hist, clt.cluster_centers_)
    # show bar chart
    plt.figure()
    plt.axis("on")
    plt.imshow(bar)
    plt.imsave(output_path, bar)
    #plt.show()

    # shows both the output and input
   # plt.show()
    # time.sleep(5)
    # plt.close("all") # does not work


# display_image()

vidcap = cv2.VideoCapture("C:\\Users\\cuppy\\Downloads\\y2mate.com - Oh my god he on xgames vine_OWl_HlyHeVc_240p.mp4")
success, image = vidcap.read()
count = 0
while success:
  cv2.imwrite("TestResults2\\frame%d.jpg" % count, image) # save frame as JPEG file
  success, image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
count = 0
while count < 10:
    display_image("TestResults2\\frame%d.jpg" % count, "TestResults2\\frame%d.jpg" % count, 3) #replace image with 3 clusters
    count += 1
    print("done")