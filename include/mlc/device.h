#ifndef MLC_DEVICE_H
#define MLC_DEVICE_H

/*
 * The device on which the tensor is stored. This can be either CPU or CUDA
 * (GPU).
 */
typedef enum {
    MLC_DEVICE_CPU,
    MLC_DEVICE_CUDA,
} mlc_device;

/*
 * Returns a string representation of the given mlc_device.
 */
static inline const char* mlc_device_str(mlc_device device) {
    switch (device) {
        case MLC_DEVICE_CPU:
            return "CPU";
        case MLC_DEVICE_CUDA:
            return "CUDA";
    }
    return "Unknown device";
}

#endif
