import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import numpy as np
import time
from multiprocessing import Process
import multiprocessing


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
    bar = np.zeros((512, 1280, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster

    for (percent, color) in zip(hist, centroids):
        # define range based on percent dominance of a color
        endX = startX + (percent * 1280)
        # create a rectangle, define the size, covert clusters to rgb, add to barchart
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 512),
                      color.astype("uint8").tolist(), -1)
        # define new beginning for next bar
        startX = endX

    return bar


def display_image(input_path, output_path, clusters, count):
    print("frame: {}".format(count))
    input_path_exact = "{}\\frame%d.jpg".format(input_path) % count
    output_path_exact = "{}\\frame%d.jpg".format(output_path) % count
    convert_image(input_path_exact, clusters)
    # generate histogram,
    hist = centroid_histogram(clt)
    # generate barchart
    bar = plot_colors(hist, clt.cluster_centers_)
    # show bar chart
    plt.figure()
    plt.axis("on")
    plt.imsave(output_path_exact, bar)
    plt.close()
    count += 2
    display_image(input_path, output_path, clusters, count)


def extract_frames(video_location, output_path):
    vidcap = cv2.VideoCapture(video_location)
    success, image = vidcap.read()
    count = 1
    while success:
        cv2.imwrite("{}\\frame%d.jpg".format(output_path) % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        print(count)
        count += 1


if __name__ == '__main__':
    #extract_frames("somewhereinthecrowd.mp4", "input")
    processes = []
    for i in range(1, 3):
        print(i)
        p = multiprocessing.Process(target=display_image, args=("input", "output", 5, i))
        p.start()
        processes.append(p)
    for process in processes:
        process.join()
