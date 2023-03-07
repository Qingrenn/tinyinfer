#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <stdlib.h>
#include <mutex>
#include <list>
#include <utility>

namespace tinyinfer {

#define TINYINFER_MALLOC_ALIGN 16

#define TINYINFER_MALLOC_OVERHEAD 64

template<typename Tp>
static inline Tp* alignPtr(Tp* ptr, int n = (int)sizeof(Tp))
{
    return (Tp*)(((size_t)ptr + n - 1) & -n);
}

static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

static inline void* fastMalloc(size_t size)
{
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + TINYINFER_MALLOC_OVERHEAD);
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, TINYINFER_MALLOC_OVERHEAD);
    adata[-1] = udata;
    return adata;
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}

class Allocator
{
public:
    virtual ~Allocator();
    virtual void* fastMalloc(size_t size) = 0;
    virtual void fastFree(void* ptr) = 0;
};

class PoolAllocator : public Allocator
{
public:
    PoolAllocator();
    ~PoolAllocator();
    PoolAllocator(const PoolAllocator&) = delete;            // forbiden copy construction
    PoolAllocator& operator=(const PoolAllocator&) = delete; // forbiden copy assignment

    void clear();
    void set_size_compare_ratio(float scr);
    void set_size_drop_threshold(size_t threshold);

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    class PoolAllocatorPrivate
    {
    public:
        std::mutex budgets_lock;
        std::mutex payouts_lock;
        unsigned int size_compare_ratio;
        size_t size_drop_threshold;
        std::list<std::pair<size_t, void*> > budgets;
        std::list<std::pair<size_t, void*> > payouts;
    };

    PoolAllocatorPrivate* const d;
};

} // namespace TINYINFER

#endif
