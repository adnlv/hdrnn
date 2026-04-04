#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "mnist.h"

int main(void)
{
    struct mnist_labels labels;
    struct mnist_images images;

    errno = 0;
    if (mnist_load_labels("assets/train-labels-idx1-ubyte", &labels) != 0) {
        perror(errno != 0 ? strerror(errno) : "Error: mnist_load_labels");
        return 1;
    }

    printf("labels.count: %u\n", labels.cnt);
    printf("labels.data[0..10]: ");
    for (size_t i = 0; i < 10; ++i) {
        printf("%u%s", labels.data[i], i == 9 ? "\n" : ", ");
    }

    errno = 0;
    if (mnist_load_images("assets/train-images-idx3-ubyte", &images) != 0) {
        perror(errno != 0 ? strerror(errno) : "Error: mnist_load_images");
        return 1;
    }

    printf("images.images_count: %u\n", images.cnt);
    printf("images.image_length: %u\n", images.len);

    mnist_free_labels(&labels);
    mnist_free_images(&images);
    return 0;
}
