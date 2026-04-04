#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MNIST_MAX_COUNT 60000
#define MNIST_IMG_SIDE_SIZE 28

/**
 * @param buf magic bytes buffer
 * @param d number of expected dimensions
 */
#define MNIST_MAGIC_MATCHES_IDX_FORMAT(buf, d)                               \
    ((buf)[0] == 0 && (buf)[1] == 0 && (buf)[2] == 8 && (buf)[3] == (d))

static uint32_t buf4_to_u32(const uint8_t *buf)
{
    return (uint32_t) buf[0] << 24 | (uint32_t) buf[1] << 16
           | (uint32_t) buf[2] << 8 | (uint32_t) buf[3];
}

int ds_load_mnist_labels(const char *filename, struct dataset *ds)
{
    FILE *file = NULL;
    uint8_t buf[4], *y = NULL;
    uint32_t c;

    file = fopen(filename, "rb");
    if (file == NULL)
        goto FAIL;

    if (fread(buf, sizeof(uint8_t), 4, file) != 4)
        goto FCLOSE;

    if (!MNIST_MAGIC_MATCHES_IDX_FORMAT(buf, 1))
        goto FCLOSE;

    if (fread(buf, sizeof(uint8_t), 4, file) != 4)
        goto FCLOSE;

    c = buf4_to_u32(buf);
    if (c == 0 || c > MNIST_MAX_COUNT)
        goto FCLOSE;

    y = malloc(sizeof(uint8_t) * c);
    if (y == NULL)
        goto FCLOSE;

    if (fread(y, sizeof(uint8_t), c, file) != (size_t) c)
        goto FREE;

    ds->c = c;
    ds->y = y;

    fclose(file);
    return 0;

FREE:
    free(y);
FCLOSE:
    fclose(file);
FAIL:
    return -1;
}

int ds_load_mnist_images(const char *filename, struct dataset *ds)
{
    FILE *file = NULL;
    uint8_t buf[4], *bytes = NULL;
    uint32_t c, n, n_w, n_h;
    size_t bytes_len;
    float *x = NULL;

    file = fopen(filename, "rb");
    if (file == NULL)
        goto FAIL;

    if (fread(buf, sizeof(uint8_t), 4, file) != 4)
        goto FCLOSE;

    if (!MNIST_MAGIC_MATCHES_IDX_FORMAT(buf, 3))
        goto FCLOSE;

    if (fread(buf, sizeof(uint8_t), 4, file) != 4)
        goto FCLOSE;

    c = buf4_to_u32(buf);
    if (c == 0 || c > MNIST_MAX_COUNT)
        goto FCLOSE;

    if (fread(buf, sizeof(uint8_t), 4, file) != 4)
        goto FCLOSE;

    n_w = buf4_to_u32(buf);
    if (n_w != MNIST_IMG_SIDE_SIZE)
        goto FCLOSE;

    if (fread(buf, sizeof(uint8_t), 4, file) != 4)
        goto FCLOSE;

    n_h = buf4_to_u32(buf);
    if (n_h != MNIST_IMG_SIDE_SIZE)
        goto FCLOSE;

    n = n_w * n_h;
    bytes_len = c * n;
    bytes = malloc(sizeof(uint8_t) * bytes_len);
    if (bytes == NULL)
        goto FCLOSE;

    if (fread(bytes, sizeof(uint8_t), bytes_len, file) != bytes_len)
        goto FREE;

    x = malloc(sizeof(float) * bytes_len);
    if (x == NULL)
        goto FREE;

    for (size_t i = 0; i < bytes_len; ++i)
        x[i] = (float) bytes[i] / 255.0f;

    ds->c = c;
    ds->n = n;
    ds->x = x;

    free(bytes);
    fclose(file);
    return 0;

FREE:
    free(bytes);
FCLOSE:
    fclose(file);
FAIL:
    return -1;
}

void ds_free(struct dataset *ds)
{
    free(ds->x);
    free(ds->y);
    memset(ds, 0, sizeof(struct dataset));
}

int ds_shuffle(struct dataset *ds)
{
    const size_t stride = sizeof(float) * ds->n;
    float *buf = NULL;

    if (ds->c <= 1)
        return 0;

    buf = malloc(sizeof(float) * ds->n);
    if (buf == NULL)
        return -1;

    for (size_t i = ds->c - 1; i > 0; --i) {
        const size_t r = rand() % (i + 1);
        float *xi = ds->x + i * ds->n;
        float *xr = ds->x + r * ds->n;
        uint8_t tmp;

        memcpy(buf, xi, stride);
        memcpy(xi, xr, stride);
        memcpy(xr, buf, stride);

        tmp = ds->y[i];
        ds->y[i] = ds->y[r];
        ds->y[r] = tmp;
    }

    free(buf);
    return 0;
}
