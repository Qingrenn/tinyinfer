#ifndef MAT_H
#define MAT_H

#include "common.h"
#include "allocator.h"
#include <cassert>

namespace tinyinfer {

class Mat
{
public:
    // constructor & destructor & assign
    Mat();
    Mat(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    Mat(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
    Mat(int w, int h, int d, size_t elemsize = 4u, Allocator* allocator = 0);
    Mat(int w, int h, int d, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    Mat(int w, size_t elemsize, int elempack, Allocator* allocator);
    Mat(int w, int h, size_t elemsize, int elempack, Allocator* allocator);
    Mat(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator);
    Mat(int w, int h, int d, int c, size_t elemsize, int elempack, Allocator* allocator);

    Mat(int w, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    Mat(int w, int h, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    Mat(int w, int h, int c, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    Mat(int w, int h, int d, int c, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    Mat(int w, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    Mat(int w, int h, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    Mat(int w, int h, int c, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    Mat(int w, int h, int d, int c, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);

    Mat(const Mat&);
    Mat& operator=(const Mat&);
    
    ~Mat();

    Mat clone(Allocator* allocator = 0) const;
    void clone_from(const tinyinfer::Mat& mat, Allocator* allocator = 0);

    void create(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    void create(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
    void create(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    void create(int w, int h, int d, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    void create(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
    void create(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
    void create(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    void create(int w, int h, int d, int c, size_t elemsize, int elempack, Allocator* allocator = 0);

    // state management
    void release();
    void addref();

    // capacity
    int total() const;
    bool empty() const;

    // reshape
    Mat reshape(int w, Allocator* allocator = 0) const;
    Mat reshape(int w, int h, Allocator* allocator = 0) const;
    Mat reshape(int w, int h, int c, Allocator* allocator = 0) const;
    Mat reshape(int w, int h, int d, int c, Allocator* allocator = 0) const;

    // element access
    Mat channel(int c);
    Mat channel_range(int c, int channels);
    Mat depth(int d);
    Mat depth_range(int d, int depths);
    float* row(int y);
    Mat row_range(int y, int rows);

    const Mat channel(int c) const;
    const Mat channel_range(int c, int channels) const;
    const Mat depth(int d) const;
    const Mat depth_range(int d, int depths) const;
    const float* row(int y) const;
    const Mat row_range(int y, int rows) const;

    template<typename T>
    T* row(int y);
    template<typename T>
    const T* row(int y) const;

    template<typename T>
    operator T*();
    template<typename T>
    operator const T*() const;

    float& operator[](size_t i);
    const float& operator[](size_t i) const;

    // modifiers
    template<typename T>
    void fill(T val);

public:
    void* data;

    Allocator* allocator;

    std::atomic<int>* refcount;

    size_t elemsize;

    int elempack;

    int dims;

    int w;
    int h;
    int d;
    int c;

    size_t cstep;
};

Mat::Mat()
    : data(0), allocator(0), refcount(0), elemsize(0), elempack(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
}

Mat::Mat(int _w, size_t _elemsize, Allocator* _allocator)
    : data(0), allocator(0), refcount(0), elemsize(0), elempack(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _elemsize, _allocator);
}

Mat::Mat(int _w, int _h, size_t _elemsize, Allocator* _allocator)
    : data(0), allocator(0), refcount(0), elemsize(0), elempack(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _allocator);
}

Mat::Mat(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
    : data(0), allocator(0), refcount(0), elemsize(0), elempack(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _allocator);
}

Mat::Mat(int _w, int _h, int _d, int _c, size_t _elemsize, Allocator* _allocator)
    : data(0), allocator(0), refcount(0), elemsize(0), elempack(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _d, _c, _elemsize, _allocator);
}

Mat::Mat(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), allocator(0), refcount(0), elemsize(0), elempack(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _elemsize, _elempack, _allocator);
}

Mat::Mat(int _w, int _h, int _d, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), allocator(0), refcount(0), elemsize(0), elempack(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _d, _c, _elemsize, _elempack, _allocator);
}

Mat::Mat(int _w, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), allocator(_allocator), refcount(0), elemsize(_elemsize), elempack(1), dims(1), w(_w), h(1), d(1), c(1)
{
    cstep = w;
}

Mat::Mat(int _w, int _h, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), allocator(_allocator), refcount(0), elemsize(_elemsize), elempack(1), dims(2), w(_w), h(_h), d(1), c(1)
{
    cstep = (size_t)w * h;
}

Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), allocator(_allocator), refcount(0), elemsize(_elemsize), elempack(1), dims(3), w(_w), h(_h), d(1), c(_c)
{
    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
}

Mat::Mat(int _w, int _h, int _d, int _c, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), allocator(_allocator), refcount(0), elemsize(_elemsize), elempack(1), dims(4), w(_w), h(_h), d(_d), c(_c)
{
    cstep = alignSize((size_t)w * h * d * elemsize, 16) / elemsize;
}

Mat::Mat(int _w, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), allocator(_allocator), refcount(0), elemsize(_elemsize), elempack(_elempack), dims(1), w(_w), h(1), d(1), c(1)
{
    cstep = w;
}

Mat::Mat(int _w, int _h, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), allocator(_allocator), refcount(0), elemsize(_elemsize), elempack(_elempack), dims(2), w(_w), h(_h), d(1), c(1)
{
    cstep = (size_t)w * h;
}

Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), allocator(_allocator), refcount(0), elemsize(_elemsize), elempack(_elempack), dims(3), w(_w), h(_h), d(1), c(_c)
{
    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
}

Mat::Mat(int _w, int _h, int _d, int _c, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), allocator(_allocator), refcount(0), elemsize(_elemsize), elempack(_elempack), dims(4), w(_w), h(_h), d(_d), c(_c)
{
    cstep = alignSize((size_t)w * h * d * elemsize, 16) / elemsize;
}

Mat::Mat(const Mat&)
{
    addref();
}

Mat& Mat::operator=(const Mat& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        ATOMIC_ADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    dims = m.dims;
    w = m.w;
    h = m.h;
    d = m.d;
    c = m.c;
    cstep = m.cstep;

    return *this;
}

Mat::~Mat()
{
    release();
}

template<typename T>
T* Mat::row(int y)
{
    return (T*)((unsigned char*)data + (size_t)w * y * elemsize);
}

template<typename T>
const T* Mat::row(int y) const
{
    return (const T*)((unsigned char*)data + (size_t)w * y * elemsize);
}

template<typename T>
Mat::operator T* ()
{
    return (T*)data;
}

template<typename T>
Mat::operator const T* () const
{
    return (T*)data;
}

template<typename T>
void Mat::fill(T val)
{
    assert(sizeof(T) == elemsize);

    int total_size = total();
    T* ptr = (T*)data;
    for (int i = 0; i < total_size; i++)
        ptr[i] = val;
}

} // namespace tinyinfer

#endif