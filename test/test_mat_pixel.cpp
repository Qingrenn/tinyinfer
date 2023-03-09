#include "mat.h"
#include "prng.h"

static struct prng_rand_t g_prng_rand_state;
#define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
#define RAND()      prng_rand(&g_prng_rand_state)

static tinyinfer::Mat RandomMat(int w, int h, int elempack)
{
    tinyinfer::Mat m(128, 128, (size_t)elempack, elempack);
    unsigned char* p = m;
    for (int i = 0; i < w * h * elempack; i++)
    {
        p[i] = RAND() % 256;
    }

    return m;
}

static int test_from_rgb()
{
    
}

int main()
{
    SRAND(1126);

    

}