import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

if __name__ == '__main__':
    images = ['./images/im_0.png',
              './images/im_1.png',
              './images/im_2.png']

    fig = plt.figure()
    ims = []
    for image in images:
        image_np = plt.imread(image)
        ims.append([plt.imshow(image_np, cmap='gray')])

    ani = animation.ArtistAnimation(fig, ims, interval=1000)

    ani.save('test.mp4')

    plt.show()
