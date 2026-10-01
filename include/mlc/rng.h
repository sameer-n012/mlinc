#ifndef MLC_RNG_H
#define MLC_RNG_H

#include <stdint.h>
#include <math.h>

typedef struct {
    uint64_t state[4];
} mlc_rng;

static inline uint64_t splitmix64(uint64_t* curr) {
    uint64_t z = (*curr += 0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

static inline void mlc_rng_seed(mlc_rng* rng, uint64_t seed) {
    rng->state[0] = splitmix64(&seed);
    rng->state[1] = splitmix64(&seed);
    rng->state[2] = splitmix64(&seed);
    rng->state[3] = splitmix64(&seed);
}

static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

static inline uint64_t mlc_rand(mlc_rng* rng) {
    uint64_t* s = rng->state;
	const uint64_t result = rotl(s[1] * 5, 7) * 9;

	const uint64_t t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 45);

	return result;
}

static inline double mlc_randu(mlc_rng* rng) {
    return (mlc_rand(rng) >> 11) * (1.0 / 9007199254740991.0);
}

static inline double mlc_randn(mlc_rng* rng) {
    double u1 = mlc_randu(rng);
    double u2 = mlc_randu(rng);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

#endif
