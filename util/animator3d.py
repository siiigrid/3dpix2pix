import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2


class MedicalImageAnimator(object):
    i = 0
    pause = False

    def __init__(
        self, data, annotation, dim=0, marker_size=[1, 1, 1], save=False,
        index=True, save_png=False
    ):
        print(type(data), data.shape)
        data[0][0][0] = np.amax(data)
        data[0][0][1] = np.amin(data)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        dims = list(range(3))
        dims.pop(dim)

        ant, = ax.plot(
            [], [], "o", markersize=2, fillstyle='full'
        )
        print(dim)

        if dim == 0:
            img = ax.imshow(data[0, :, :], 'gray')

        elif dim == 1:
            print("dim1",data[:,0,:].shape)
            img = ax.imshow(data[:, 0, :], 'gray')

        elif dim == 2:
            print("dim3",data[:,:,0].shape)
            img = ax.imshow(data[:, :, 0], 'gray')

        if index:
            time_text = ax.text(
                0.03, 0.95, '',
                color='red',
                fontsize=24,
                horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes
            )
        else:
            time_text = None

        fig.canvas.mpl_connect('key_press_event', self.onkey)

        self.data = data
        self.annotation = np.array(annotation)
        self.marker_size = marker_size
        self.dim = dim
        self.dims = dims
        self.fig = fig
        self.ax = ax
        self.ant = ant
        self.img = img
        self.time_text = time_text
        self.save = save
        self.save_png = save_png

    def onkey(self, event):
        key = event.key
        if key == 'a':
            self.pause ^= True
        return

    def update(self, data):
        plt.axis('off')

        """if self.save_png:
            frame_filename = f"frame_{self.i:04d}.png"
            self.fig.savefig(frame_filename, transparent = True, dpi = 90)"""

        if not self.pause:
            self.img.set_array(data)
            

            markers_x = []
            markers_y = []
            if self.annotation.size > 0:
                for index, location in enumerate(self.annotation[:, self.dim]):
                    if location - self.marker_size[self.dim] < self.i and \
                       self.i < location + self.marker_size[self.dim]:
                        markers_x.append(self.annotation[index, self.dims[1]])
                        markers_y.append(self.annotation[index, self.dims[0]])

            self.ant.set_data(markers_x, markers_y)
            if self.time_text:
                self.time_text.set_text(
                    'index = {}'.format(self.i % self.data.shape[self.dim])
                )

        return self.img, self.ant, self.time_text

    def generate_data(self):
        print("dim ", self.dim)
        if self.dim == 0:
            data = self.data
        elif self.dim == 1:
            data = self.data.transpose(1, 0, 2)
        elif self.dim == 2:
            data = self.data.transpose(2, 0, 1)

        self.i = -1
        while self.i < data.shape[0] - 1:
            if not self.pause:
                self.i += 1
            yield data[self.i]

    def run(self, save_dir):
        plt.tight_layout()
        animate = animation.FuncAnimation(
            self.fig, self.update, self.generate_data,
            interval=50, repeat=True,
            save_count=self.data.shape[self.dim]
        )
        if self.save:
            animate.save(
                save_dir, writer='imagemagick', fps=15
            )
        else:
            plt.show()
        return animate

if __name__ == "__main__":
    image_file = "images/WbNG ZHj NbN.9.1681.npy"
    image_file = "datasets/nodules/heatmaps/real/1.3.6.1.4.1.14519.5.2.1.6279.6001.220596530836092324070084384692.npz"
    image = np.load(image_file)['a']
    image[0][0][0] = 1

    animator = MedicalImageAnimator(image, [], 0, save=False)
    animate = animator.run()
