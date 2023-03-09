#include "mat.h"
#include "common.h"

namespace tinyinfer {

static int from_rgb(const unsigned char* rgb, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 3, 4u, allocator);
    if (m.empty())
        return -1;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
        int remain = w;
        for (; remain > 0; remain--)
        {
            *ptr0 = rgb[0];
            *ptr1 = rgb[1];
            *ptr2 = rgb[2];

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++; 
        }
        rgb += wgap;
    }
    return 0;
}

static void to_rgb(const Mat& m, unsigned char* rgb, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

        int remain = w;
        for (; remain > 0; remain--)
        {
            rgb[0] = SATURATE_CAST_UCHAR(*ptr0);
            rgb[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgb[2] = SATURATE_CAST_UCHAR(*ptr2);

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }
        rgb += wgap;
    }
}


static int from_gray(const unsigned char* gray, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 1, 4u, allocator);
    if (m.empty())
        return -1;

    const int wgap = stride - w;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y = 0; y < h; y++)
    {
        int remain = w;
        for (; remain > 0; remain--)
        {
            *ptr = *gray;

            gray++;
            ptr++;
        }
        gray += wgap;
    }

    return 0;
}

static void to_gray(const Mat& m, unsigned char* gray, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr = m;

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

        int remain = w;
        for (; remain > 0; remain--)
        {
            *gray = SATURATE_CAST_UCHAR(*ptr);

            gray++;
            ptr++;
        }
        gray += wgap;
    }
}

Mat Mat::from_pixels(const unsigned char* pixels, int type, int w, int h, Allocator* allocator)
{
    int type_from = type & PIXEL_FORMAT_MASK;
    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 3, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 1, allocator);
    }
    TINYINFER_LOG("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels(const unsigned char* pixels, int type, int w, int h, int stride, Allocator* allocator)
{
    Mat m;
    if (type & PIXEL_CONVERT_MASK)
    {
        TINYINFER_LOG("undeveloped fuction %d", type);
    }else 
    {
        if (type == PIXEL_RGB || type == PIXEL_BGR)
            from_rgb(pixels, w, h, stride, m, allocator);

        if (type == PIXEL_GRAY)
            from_gray(pixels, w, h, stride, m, allocator);
    }

    return m;
}

void Mat::to_pixels(unsigned char* pixels, int type) const
{

}

void Mat::to_pixels(unsigned char* pixels, int type, int stride) const
{

}

}