#ifndef MLC_RNG_H
#define MLC_RNG_H

#include <stdint.h>
#include <math.h>

#define MLC_PI 3.14159265358979323846

typedef struct {
    uint64_t state[4];
} mlc_rng;

static inline uint64_t mlc_rng_splitmix64(uint64_t* curr) {
    uint64_t z = (*curr += 0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

static inline void mlc_rng_seed(mlc_rng* rng, uint64_t seed) {
    rng->state[0] = mlc_rng_splitmix64(&seed);
    rng->state[1] = mlc_rng_splitmix64(&seed);
    rng->state[2] = mlc_rng_splitmix64(&seed);
    rng->state[3] = mlc_rng_splitmix64(&seed);
}

static inline uint64_t mlc_rng_rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

static inline uint64_t mlc_rng_next(mlc_rng* rng) {
    uint64_t* s = rng->state;
	const uint64_t result = mlc_rng_rotl(s[1] * 5, 7) * 9;

	const uint64_t t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = mlc_rng_rotl(s[3], 45);

	return result;
}

static inline double mlc_randu(mlc_rng* rng) {
    return (mlc_rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

static inline double mlc_randn(mlc_rng* rng) {
    double u1 = mlc_randu(rng);
    while (u1 == 0.0) {
        u1 = mlc_randu(rng);
    }
    double u2 = mlc_randu(rng);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * MLC_PI * u2);
}

#endif
