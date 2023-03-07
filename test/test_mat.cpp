#include "mat.h"
#include <stdlib.h>

static int test_mat_init1()
{
    tinyinfer::Mat mat(16);
    tinyinfer::Mat mat1(16, 16);
    tinyinfer::Mat mat2(16, 16, 3);
    tinyinfer::Mat mat3(16, 16, 16, 3);
    return 0;
}

static int test_mat_clone()
{
    tinyinfer::Mat mat(16, 16, 16, 3);
    tinyinfer::Mat mat2;
    mat2.clone_from(mat);
    return 0;
}

static int test_mat_reshape()
{
    tinyinfer::Mat mat(16, 16, 16, 3);
    printf("%d %d %d %d %d\n", mat.dims, mat.w, mat.h, mat.d, mat.c);
    mat.reshape(16 * 16 * 16 * 3);
    mat.reshape(16 * 16, 16 * 3);
    mat.reshape(16, 16, 48);
    mat.reshape(16, 16, 3, 16);
    return 0;
}

static int test_mat_elemaccess()
{
    tinyinfer::Mat mat(16, 16, 16, 3);
    tinyinfer::Mat m;
    m = mat.channel(0);
    m = mat.channel_range(0, 2);
    m = mat.depth(0);
    m = mat.depth_range(0, 8);
    m = mat.row_range(0, 8);

    float* p;
    p = mat.row(0);
    
    return 0;
}

static int test_mat_fill()
{
    tinyinfer::Mat m(16);
    m.fill(1);
    m.fill(1.f);
    m.fill<float>(1.f);
    return 0;
}

int main()
{
    return 0 || test_mat_init1() \
             || test_mat_clone() \
             || test_mat_reshape()
             || test_mat_elemaccess() \
             || test_mat_fill();
}