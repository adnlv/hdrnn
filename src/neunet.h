#ifndef HDRNN_NEUNET_H
#define HDRNN_NEUNET_H

#include <stdint.h>

struct nn_layer
{
    uint32_t n_in; // Number of inputs
    uint32_t n_out; // Number of outputs
    float *w; // Weight matrix
    float *b; // Bias vector
    float *z; // Weighted sums before activation
    float *a; // Activated outputs
};

extern int nn_init_layer(uint32_t n_in, uint32_t n_out,
                         struct nn_layer *layer);

extern void nn_free_layer(struct nn_layer *layer);

#endif // HDRNN_NEUNET_H
