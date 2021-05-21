from io import BytesIO

import scipy.misc
import tensorflow as tf


class Logger(object):

    def __init__(self, log_dir):
        return
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        return
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()


    def image_summary(self, tag, image, step):
        return
        s = BytesIO()
        scipy.misc.toimage(image).save(s, format="png")

        # Create an Image object
        img_sum = tf.Summary.Image(
            encoded_image_string=s.getvalue(),
            height=image.shape[0],
            width=image.shape[1],
        )

        # Create and write Summary
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()


    def image_list_summary(self, tag, images, step):
        return
        if len(images) == 0:
            return
        img_summaries = []
        for i, img in enumerate(images):
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            # img_sum = tf.Summary.Image(
            #     encoded_image_string=s.getvalue(),
            #     height=img.shape[0],
            #     width=img.shape[1],
            # )

            # Create a Summary value
            # img_summaries.append(
            #     tf.Summary.Value(tag="{}/{}".format(tag, i), image=img_sum)
            # )

        # Create and write Summary
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

