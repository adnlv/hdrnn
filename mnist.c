#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>

static uint32_t bytes_buf_to_u32(const uint8_t *const buf)
{
    return (uint32_t) buf[0] << 24 | (uint32_t) buf[1] << 16
           | (uint32_t) buf[2] << 8 | (uint32_t) buf[3];
}

int mnist_load_labels(const char *const filename, mnist_labels_t *const labels)
{
    FILE *file;
    size_t read;
    uint8_t bytes_buf[4], *data;
    uint32_t magic, count;

    if (filename == NULL || labels == NULL) {
        goto FAIL;
    }

    labels->count = 0;
    labels->data = NULL;

    file = fopen(filename, "rb");
    if (file == NULL) {
        goto FAIL;
    }

    read = fread(bytes_buf, sizeof(uint8_t), 4, file);
    if (read != 4) {
        goto FCLOSE;
    }

    magic = bytes_buf_to_u32(bytes_buf);
    if (magic != MNIST_LABELS_MAGIC) {
        goto FCLOSE;
    }

    read = fread(bytes_buf, sizeof(uint8_t), 4, file);
    if (read != 4) {
        goto FCLOSE;
    }

    count = bytes_buf_to_u32(bytes_buf);
    if (count == 0 || count > MNIST_LABELS_MAX_COUNT) {
        goto FCLOSE;
    }

    data = malloc(sizeof(uint8_t) * count);
    if (data == NULL) {
        goto FCLOSE;
    }

    read = fread(data, sizeof(uint8_t), count, file);
    if (read != (size_t) count) {
        goto FREE_DATA;
    }

    labels->count = count;
    labels->data = data;

    fclose(file);
    return 0;

FREE_DATA:
    free(data);
FCLOSE:
    fclose(file);
FAIL:
    return -1;
}

void mnist_free_labels(mnist_labels_t *const labels)
{
    if (labels == NULL) {
        return;
    }

    free(labels->data);

    labels->data = NULL;
    labels->count = 0;
}

int mnist_load_images(const char *const filename, mnist_images_t *const images)
{
    FILE *file;
    size_t read;
    uint8_t bytes_buf[4], *data;
    uint32_t magic, images_count, image_length, image_width, image_height,
            count;

    if (filename == NULL || images == NULL) {
        goto FAIL;
    }

    images->images_count = 0;
    images->image_length = 0;
    images->data = NULL;

    file = fopen(filename, "rb");
    if (file == NULL) {
        goto FAIL;
    }

    // Read magic number
    read = fread(bytes_buf, sizeof(uint8_t), 4, file);
    if (read != 4) {
        goto FCLOSE;
    }

    magic = bytes_buf_to_u32(bytes_buf);
    if (magic != MNIST_IMAGES_MAGIC) {
        goto FCLOSE;
    }

    // Read total number of images
    read = fread(bytes_buf, sizeof(uint8_t), 4, file);
    if (read != 4) {
        goto FCLOSE;
    }

    images_count = bytes_buf_to_u32(bytes_buf);
    if (images_count == 0 || images_count > MNIST_IMAGES_MAX_COUNT) {
        goto FCLOSE;
    }

    // Read image width
    read = fread(bytes_buf, sizeof(uint8_t), 4, file);
    if (read != 4) {
        goto FCLOSE;
    }

    image_width = bytes_buf_to_u32(bytes_buf);
    if (image_width != MNIST_IMAGE_WIDTH) {
        goto FCLOSE;
    }

    // Read image height
    read = fread(bytes_buf, sizeof(uint8_t), 4, file);
    if (read != 4) {
        goto FCLOSE;
    }

    image_height = bytes_buf_to_u32(bytes_buf);
    if (image_height != MNIST_IMAGE_HEIGHT) {
        goto FCLOSE;
    }

    image_length = image_width * image_height;
    if (image_length != MNIST_IMAGE_LENGTH) {
        goto FCLOSE;
    }

    // Read data
    count = images_count * image_length;
    data = malloc(sizeof(uint8_t) * count);
    if (data == NULL) {
        goto FCLOSE;
    }

    read = fread(data, sizeof(uint8_t), count, file);
    if (read != (size_t) count) {
        goto FREE_DATA;
    }

    images->images_count = images_count;
    images->image_length = image_length;
    images->data = data;

    fclose(file);
    return 0;

FREE_DATA:
    free(data);
FCLOSE:
    fclose(file);
FAIL:
    return -1;
}

void mnist_free_images(mnist_images_t *const images)
{
    if (images == NULL) {
        return;
    }

    free(images->data);

    images->data = NULL;
    images->images_count = 0;
    images->image_length = 0;
}
