/*
 * test_storage.c - tests for mlc_storage (include/mlc/storage.h).
 *
 * Note: ASan on macOS does not detect leaks. To check that release() frees the
 * storage, run the binary with `leaks --atExit -- build/debug/tests/test_storage`
 * (macOS) or under ASan/valgrind on Linux.
 */
#include <stdint.h>

#include "mlc/storage.h"
#include "mlc_test.h"

MLC_TEST(storage_new_sets_fields) {
    mlc_storage* s = mlc_storage_new(100, MLC_DEVICE_CPU);
    MLC_ASSERT(s != NULL);
    MLC_ASSERT(s->data != NULL);
    MLC_ASSERT_EQ_INT(s->size, 100);
    MLC_ASSERT_EQ_INT(s->ref_count, 1);
    MLC_ASSERT(s->device == MLC_DEVICE_CPU);
    MLC_ASSERT_MSG((uintptr_t)s->data % 64 == 0, "data is not 64-byte aligned");
    mlc_storage_release(s);
}

MLC_TEST(storage_size_not_multiple_of_alignment) {
    /* aligned_alloc needs a size that is a multiple of the alignment; the
     * allocator must round up. 3 bytes is the smallest bad case. */
    mlc_storage* s = mlc_storage_new(3, MLC_DEVICE_CPU);
    MLC_ASSERT(s != NULL && s->data != NULL);
    MLC_ASSERT_EQ_INT(s->size, 3);
    ((unsigned char*)s->data)[2] = 0xAB; /* ASan: must be writable */
    mlc_storage_release(s);
}

MLC_TEST(storage_zero_size_has_valid_pointer) {
    mlc_storage* s = mlc_storage_new(0, MLC_DEVICE_CPU);
    MLC_ASSERT(s != NULL);
    MLC_ASSERT(s->data != NULL);
    MLC_ASSERT_EQ_INT(s->size, 0);
    mlc_storage_release(s);
}

MLC_TEST(storage_retain_release_counts) {
    mlc_storage* s = mlc_storage_new(16, MLC_DEVICE_CPU);
    MLC_ASSERT(s != NULL);
    mlc_storage_retain(s);
    MLC_ASSERT_EQ_INT(s->ref_count, 2);
    mlc_storage_retain(s);
    MLC_ASSERT_EQ_INT(s->ref_count, 3);
    mlc_storage_release(s);
    MLC_ASSERT_EQ_INT(s->ref_count, 2);
    mlc_storage_release(s);
    MLC_ASSERT_EQ_INT(s->ref_count, 1);
    /* The data must still be usable while one reference is left. */
    ((float*)s->data)[3] = 1.5f;
    mlc_storage_release(s); /* frees; ASan reports a use-after-free if not */
}

MLC_TEST(storage_release_null_is_noop) { mlc_storage_release(NULL); }
