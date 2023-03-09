#include "allocator.h"
#include "common.h"

namespace tinyinfer {

Allocator::~Allocator()
{
}

PoolAllocator::PoolAllocator()
    : Allocator(), d(new PoolAllocatorPrivate())
{
    d->size_compare_ratio = 0;
    d->size_drop_threshold = 10;
}

PoolAllocator::~PoolAllocator()
{
    clear();
    if (!d->payouts.empty())
    {
        std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
        for (; it != d->payouts.end(); it++)
        {
            void* ptr = it->second;
            TINYINFER_LOG("%p still in use", ptr);
        }
    }

    delete d;
}

void PoolAllocator::clear()
{
    d->budgets_lock.lock();

    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin();
    for (; it != d->budgets.end(); it++)
    {
        tinyinfer::fastFree(it->second);
    }

    d->budgets.clear();

    d->budgets_lock.unlock();
}

void PoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr <= 0.f || scr > 1.f)
    {
        TINYINFER_LOG("invalid size compare ratio %f", scr);
    }

    d->size_compare_ratio = (unsigned int)(scr * 256);
}

void PoolAllocator::set_size_drop_threshold(size_t threshold)
{
    d->size_drop_threshold = threshold;
}

void* PoolAllocator::fastMalloc(size_t size)
{
    d->budgets_lock.lock();
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin(),
                                                   it_min = d->budgets.begin(),
                                                   it_max = d->budgets.begin();
    for (; it != d->budgets.end(); it++)
    {
        size_t block_size = it->first;
        if (block_size >= size && (block_size * d->size_compare_ratio >> 8) <= size)
        {
            void* ptr = it->second;
            d->budgets.erase(it);
            d->budgets_lock.unlock();

            d->payouts_lock.lock();
            d->payouts.push_back(std::make_pair(block_size, ptr));
            d->payouts_lock.unlock();
            return ptr;
        }

        if (block_size < it_min->first)
        {
            it_min = it;
        }
        if (block_size > it_max->first)
        {
            it_max = it;
        }
    }

    if (d->budgets.size() >= d->size_drop_threshold)
    {
        if (it_max->first < size)
        {
            tinyinfer::fastFree(it_min->second);
            d->budgets.erase(it_min);
        }
        else if (it_min->first > size)
        {
            tinyinfer::fastFree(it_max->second);
            d->budgets.erase(it_max);
        }
    }

    d->budgets_lock.unlock();

    void* ptr = tinyinfer::fastMalloc(size);

    d->payouts_lock.lock();
    d->payouts.push_back(std::make_pair(size, ptr));
    d->payouts_lock.unlock();

    return ptr;
}

void PoolAllocator::fastFree(void* ptr)
{
    d->payouts_lock.lock();

    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin();
    for (; it != d->budgets.end(); it++)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            d->payouts.erase(it);
            d->payouts_lock.unlock();

            d->budgets_lock.lock();
            d->budgets.push_back(std::make_pair(size, ptr));
            d->budgets_lock.unlock();

            return;
        }
    }

    d->payouts_lock.unlock();
    TINYINFER_LOG("FATAL ERROR! pool allocator get wild %p", ptr);
    tinyinfer::fastFree(ptr);
}

} // namespace tinyinfer
