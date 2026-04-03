#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>

uint32_t bytes_buf_to_u32(const uint8_t *const buf) {
    return (uint32_t) buf[0] << 24 | (uint32_t) buf[1] << 16
           | (uint32_t) buf[2] << 8 | (uint32_t) buf[3];
}

int mnist_load_labels(const char *const filename,
                      mnist_labels_t *const labels) {
    FILE *file;
    size_t read;
    uint8_t bytes_buf[4];
    uint32_t magic, count;
    uint8_t *data;

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
