#include "common.h"
#include "mat.h"
#include "string.h"

namespace tinyinfer {

Mat Mat::clone(Allocator* _allocator) const
{
    if (empty())
        return Mat();
    
    Mat m;
    if (dims == 1)
        m.create(w, elemsize, elempack, _allocator);
    else if (dims == 2)
        m.create(w, h, elemsize, elempack, _allocator);
    else if (dims == 3)
        m.create(w, h, c, elemsize, elempack, _allocator);
    else if (dims == 4)
        m.create(w, h, d, c, elemsize, elempack, _allocator);

    if (total() > 0)
    {
        if (cstep == m.cstep)
            memcpy(m.data, data, total() * elemsize);
        else
        {
            size_t size = (size_t) w * h * d * elemsize;
            for (int i = 0; i < c; i++)
            {
                memcpy(m.channel(i), channel(i), size);
            }
        }
    }
    
    return m;
}

void Mat::clone_from(const Mat& mat, Allocator* _allocator)
{
    *this = mat.clone(_allocator);
}

void Mat::create(int _w, size_t _elemsize, Allocator* _allocator)
{
    create(_w, _elemsize, 0, _allocator);
}

void Mat::create(int _w, int _h, size_t _elemsize, Allocator* _allocator)
{
    create(_w, _h, _elemsize, 0, _allocator);
}

void Mat::create(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
{
    create(_w, _h, _c, _elemsize, 0, _allocator);
}

void Mat::create(int _w, int _h, int _d, int _c, size_t _elemsize, Allocator* _allocator)
{
    create(_w, _h, _d, _c, _elemsize, 0, _allocator);
}

void Mat::create(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    d = 1;
    c = 1;

    cstep = w;
    
    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        size_t allocated_size = totalsize + (int)sizeof(*refcount);
        if (allocator)
            data = allocator->fastMalloc(allocated_size);
        else
            data = tinyinfer::fastMalloc(allocated_size);
    }

    if (data)
    {
        refcount = (std::atomic<int>*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    d = 1;
    c = 1;

    cstep = (size_t)w * h;
    
    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        size_t allocated_size = totalsize + (int)sizeof(*refcount);
        if (allocator)
            data = allocator->fastMalloc(allocated_size);
        else
            data = tinyinfer::fastMalloc(allocated_size);
    }

    if (data)
    {
        refcount = (std::atomic<int>*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    d = 1;
    c = _c;

    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
    
    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        size_t allocated_size = totalsize + (int)sizeof(*refcount);
        if (allocator)
            data = allocator->fastMalloc(allocated_size);
        else
            data = tinyinfer::fastMalloc(allocated_size);
    }

    if (data)
    {
        refcount = (std::atomic<int>*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _d, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 4;
    w = _w;
    h = _h;
    d = _d;
    c = _c;

    cstep = alignSize((size_t)w * h * d * elemsize, 16) / elemsize;
    
    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        size_t allocated_size = totalsize + (int)sizeof(*refcount);
        if (allocator)
            data = allocator->fastMalloc(allocated_size);
        else
            data = tinyinfer::fastMalloc(allocated_size);
    }

    if (data)
    {
        refcount = (std::atomic<int>*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::release()
{
    if (refcount && ATOMIC_ADD(refcount, -1) == 1)
    {
        if (allocator)
            allocator->fastFree(data);
        else
            fastFree(data);
    }

    refcount = 0;
    data = 0;
    elemsize = 0;
    elempack = 0;
    dims = 0;
    w = 0;
    h = 0;
    d = 0;
    c = 0;
    cstep = 0;
}

void Mat::addref()
{
    if (refcount)
        ATOMIC_ADD(refcount, 1);
}

int Mat::total() const
{
    return cstep * c;
}

bool Mat::empty() const
{
    return data == 0 || total() == 0;
}

/**
 * reshape
*/
Mat Mat::reshape(int _w, Allocator* _allocator) const
{
    if (w * h * d * c != _w)
        return Mat();
    
    if (dims >= 3 && cstep != (size_t) w * h * d)
    {
        Mat m;
        m.create(_w, elemsize, elempack, _allocator);
        for (int i = 0; i < c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + (size_t)i * w * h * d * elemsize;
            memcpy(mptr, ptr,  (size_t)w * h * d * elemsize);
        }
        
        return m;
    }

    Mat m = *this;
    m.dims = 1;
    m.w = _w;
    m.h = 1;
    m.d = 1;
    m.c = 1;
    m.cstep = _w;
    
    return m;
}

Mat Mat::reshape(int _w, int _h, Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h)
        return Mat();
    
    if (dims >= 3 && cstep != (size_t) w * h * d)
    {
        Mat m;
        m.create(_w, _h, elemsize, elempack, _allocator);
        for (int i = 0; i < c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + (size_t)i * w * h * d * elemsize;
            memcpy(mptr, ptr,  (size_t)w * h * d * elemsize);
        }
        
        return m;
    }

    Mat m = *this;
    m.dims = 1;
    m.w = _w;
    m.h = _h;
    m.d = 1;
    m.c = 1;
    m.cstep = (size_t)_w * _h;
    
    return m;
}

Mat Mat::reshape(int _w, int _h, int _c, Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h * _c)
        return Mat();
    
    if (dims < 3)
    {
        if ((size_t)_w * _h != alignSize((size_t)_w * _h * elemsize, 16) / elemsize)
        {
            Mat m;
            m.create(_w, _h, _c, elemsize, elempack, _allocator);
            for (int i = 0; i < _c; i++)
            {
                const void* ptr = (unsigned char*)data + (size_t)i * _w * _h * elemsize;
                void* mptr = (unsigned char*)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr,  (size_t)_w * _h * elemsize);
            }
            return m;
        }
    }
    else if (c != _c)
    {
        Mat tmp = reshape(_w * _h * _c, _allocator);
        return tmp.reshape(_w, _h, _c, _allocator);
    }

    Mat m = *this;
    m.dims = 3;
    m.w = _w;
    m.h = _h;
    m.d = 1;
    m.c = _c;
    m.cstep = alignSize((size_t)_w * _h * elemsize, 16) / elemsize;
    
    return m;
}

Mat Mat::reshape(int _w, int _h, int _d, int _c, Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h * _d * _c)
        return Mat();

    if (dims < 3)
    {
        if ((size_t)_w * _h * _d != alignSize((size_t)_w * _h * _d * elemsize, 16) / elemsize)
        {
            Mat m;
            m.create(_w, _h, _d, _c, elemsize, elempack, _allocator);

            // align channel
            for (int i = 0; i < _c; i++)
            {
                const void* ptr = (unsigned char*)data + (size_t)i * _w * _h * _d * elemsize;
                void* mptr = (unsigned char*)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, (size_t)_w * _h * _d * elemsize);
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        Mat tmp = reshape(_w * _h * _d * _c, _allocator);
        return tmp.reshape(_w, _h, _d, _c, _allocator);
    }

    Mat m = *this;

    m.dims = 4;
    m.w = _w;
    m.h = _h;
    m.d = _d;
    m.c = _c;

    m.cstep = alignSize((size_t)_w * _h * _d * elemsize, 16) / elemsize;

    return m;
}

/**
 * element access
*/

Mat Mat::channel(int _c)
{
    Mat m(w, h, d, (unsigned char*)data + cstep * c * elemsize, elemsize, elempack, allocator);
    m.dims = dims - 1;
    if (dims == 4)
        m.cstep = (size_t)w * h;
    return m;
}

Mat Mat::channel_range(int c, int channels)
{
    Mat m(w, h, d, channels, (unsigned char*)data + cstep * c * elemsize, elemsize, elempack, allocator);
    m.dims = dims;
    return m;
}

Mat Mat::depth(int d)
{
    return Mat(w, h, (unsigned char*)data + (size_t)w * h * d * elemsize, elemsize, elempack, allocator);
}

Mat Mat::depth_range(int d, int depths)
{
    Mat m(w, h, depths, (unsigned char*)data + (size_t)w * h * d * elemsize, elemsize, elempack, allocator);
    m.cstep = (size_t)w * h;
    return m;
}

float* Mat::row(int y)
{
    return row<float>(y);
}

Mat Mat::row_range(int y, int rows)
{
    return Mat(w, rows, (unsigned char*)data + (size_t)w * y * elemsize);
}

const Mat Mat::channel(int _c) const
{
    Mat m(w, h, d, (unsigned char*)data + cstep * c * elemsize, elemsize, elempack, allocator);
    m.dims = dims - 1;
    if (dims == 4)
        m.cstep = (size_t)w * h;
    return m;
}

const Mat Mat::channel_range(int c, int channels) const
{
    Mat m(w, h, d, channels, (unsigned char*)data + cstep * c * elemsize, elemsize, elempack, allocator);
    m.dims = dims;
    return m;
}

const Mat Mat::depth(int d) const
{
    return Mat(w, h, (unsigned char*)data + (size_t)w * h * d * elemsize, elemsize, elempack, allocator);
}

const Mat Mat::depth_range(int d, int depths) const
{
    Mat m(w, h, depths, (unsigned char*)data + (size_t)w * h * d * elemsize, elemsize, elempack, allocator);
    m.cstep = (size_t)w * h;
    return m;
}

const float* Mat::row(int y) const
{
    return row<float>(y);
}

const Mat Mat::row_range(int y, int rows) const
{
    return Mat(w, rows, (unsigned char*)data + (size_t)w * y * elemsize);
}

float& Mat::operator[](size_t i)
{
    return ((float*)data)[i];
}

const float& Mat::operator[](size_t i) const
{
    return ((const float*)data)[i];
}

} // namespace tinyinfer