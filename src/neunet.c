#include "neunet.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Kaiming He's uniform distribution.
 * @see https://en.wikipedia.org/wiki/Weight_initialization#He_initialization
 */
static float he_init(uint32_t n_in) { return sqrtf(6.0f / (float) n_in); }

int nn_init_layer(uint32_t n_in, uint32_t n_out, struct nn_layer *layer)
{
    // Required to free NULL pointers safely
    memset(layer, 0, sizeof(struct nn_layer));

    layer->w = calloc(n_in * n_out, sizeof(float));
    if (layer->w == NULL)
        goto FAIL;

    for (uint32_t i = 0; i < n_in * n_out; ++i) {
        // Inspired by https://stackoverflow.com/a/13409133
        const float r = (float) rand() / (float) RAND_MAX * 2.0f - 1.0f;
        layer->w[i] = r * he_init(n_in);
    }

    layer->b = calloc(n_out, sizeof(float));
    if (layer->b == NULL)
        goto FAIL;

    layer->z = calloc(n_out, sizeof(float));
    if (layer->z == NULL)
        goto FAIL;

    layer->a = calloc(n_out, sizeof(float));
    if (layer->a == NULL)
        goto FAIL;

    layer->n_in = n_in;
    layer->n_out = n_out;
    return 0;

FAIL:
    free(layer->w);
    free(layer->b);
    free(layer->z);
    free(layer->a);
    memset(layer, 0, sizeof(struct nn_layer));
    return -1;
}

void nn_free_layer(struct nn_layer *layer)
{
    free(layer->w);
    free(layer->b);
    free(layer->z);
    free(layer->a);
    memset(layer, 0, sizeof(struct nn_layer));
}
