#ifndef HDRNN_MNIST_H
#define HDRNN_MNIST_H

#include <stdint.h>

#define MNIST_LABELS_MAGIC 0x00000801
#define MNIST_LABELS_MAX_COUNT 60000

#define MNIST_IMAGES_MAGIC 0x00000803
#define MNIST_IMAGES_MAX_COUNT 60000
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_LENGTH 784

struct mnist_labels
{
    /* Count: Number of labels stored in the dataset */
    uint32_t cnt;

    /* Data: Array of digits from 0 to 9 inclusively */
    uint8_t *data;
};

struct mnist_images
{
    /* Count: Number of images stored in the dataset */
    uint32_t cnt;

    /* Length: Individual image length (width * height) */
    uint32_t len;

    /* Data: Raw RGB pixel data in bytes */
    uint8_t *data;
};

/* Read the file with a specified filename and write a result to labels */
int mnist_load_labels(const char *filename, struct mnist_labels *labels);

/* Free resources allocated for storing labels */
void mnist_free_labels(struct mnist_labels *labels);

/* Read the file with a specified filename and write a result to images */
int mnist_load_images(const char *filename, struct mnist_images *images);

/* Free resources allocated for storing images */
void mnist_free_images(struct mnist_images *images);

#endif // HDRNN_MNIST_H
