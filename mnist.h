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

typedef struct {
    uint32_t count;
    uint8_t *data;
} mnist_labels_t;

typedef struct {
    uint32_t images_count;
    uint32_t image_length;
    uint8_t *data;
} mnist_images_t;

int mnist_load_labels(const char *filename, mnist_labels_t *labels);
void mnist_free_labels(mnist_labels_t *labels);

int mnist_load_images(const char *filename, mnist_images_t *images);
void mnist_free_images(mnist_images_t *images);

#endif // HDRNN_MNIST_H
