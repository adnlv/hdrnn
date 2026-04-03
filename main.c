#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "mnist.h"

int main(void) {
    mnist_labels_t labels;
    mnist_images_t images;

    errno = 0;
    if (mnist_load_labels("assets/train-labels-idx1-ubyte", &labels) != 0) {
        perror(errno != 0 ? strerror(errno) : "Error: mnist_load_labels");
        return 1;
    }

    printf("labels.count: %u\n", labels.count);
    printf("labels.data[0..10]: ");
    for (size_t i = 0; i < 10; ++i) {
        printf("%u%s", labels.data[i], i == 9 ? "\n" : ", ");
    }

    errno = 0;
    if (mnist_load_images("assets/train-images-idx3-ubyte", &images) != 0) {
        perror(errno != 0 ? strerror(errno) : "Error: mnist_load_images");
        return 1;
    }

    printf("images.images_count: %u\n", images.images_count);
    printf("images.image_length: %u\n", images.image_length);

    mnist_free_labels(&labels);
    mnist_free_images(&images);
    return 0;
}
