#include <cstring>
#include "mat.h"
#include "prng.h"

static struct prng_rand_t g_prng_rand_state;
#define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
#define RAND()      prng_rand(&g_prng_rand_state)

static tinyinfer::Mat RandomMat(int w, int h, int elempack)
{
    tinyinfer::Mat m(w, h, (size_t)elempack, elempack);
    unsigned char* p = m;
    for (int i = 0; i < w * h * elempack; i++)
    {
        p[i] = RAND() % 256;
    }

    return m;
}

static int test_mat_from_to_rgb(int w, int h)
{
    tinyinfer::Mat a = RandomMat(w, h, 3);
    tinyinfer::Mat m = tinyinfer::Mat::from_pixels(a, tinyinfer::Mat::PIXEL_RGB, w, h);
    tinyinfer::Mat b(w, h, (size_t)3u, 3);
    m.to_pixels(b, tinyinfer::Mat::PIXEL_RGB);
    if ( memcmp(a, b, w * h * 3) != 0)
    {
        fprintf(stderr, "test_mat_from_to_rgb failed w=%d h=%d pixel_type=%d\n", w, h, tinyinfer::Mat::PIXEL_RGB);
        return -1;
    }
    return 0;
}

static int test_mat_from_to_gray(int w, int h)
{
    tinyinfer::Mat a = RandomMat(w, h, 1);
    tinyinfer::Mat m = tinyinfer::Mat::from_pixels(a, tinyinfer::Mat::PIXEL_GRAY, w, h);
    tinyinfer::Mat b(w, h, (size_t)1u, 1);
    m.to_pixels(b, tinyinfer::Mat::PIXEL_GRAY);
    if ( memcmp(a, b, w * h * 1) != 0)
    {
        fprintf(stderr, "test_mat_from_to_gray failed w=%d h=%d pixel_type=%d\n", w, h, tinyinfer::Mat::PIXEL_GRAY);
        return -1;
    }
    return 0;
}

int main()
{
    SRAND(1126);

    return 0 || test_mat_from_to_rgb(16, 16)
             || test_mat_from_to_gray(16, 16);

}