#ifndef HDRNN_MNIST_H
#define HDRNN_MNIST_H

#include <stdint.h>

#define MNIST_LABELS_MAGIC 0x00000801
#define MNIST_LABELS_MAX_COUNT 60000

typedef struct {
    uint32_t count;
    uint8_t *data;
} mnist_labels_t;

int mnist_load_labels(const char *filename, mnist_labels_t *labels);

#endif // HDRNN_MNIST_H
