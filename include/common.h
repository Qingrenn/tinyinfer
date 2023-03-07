#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <atomic>
#define TINYINFER_LOG(...)               \
    do                                  \
    {                                   \
        fprintf(stderr, ##__VA_ARGS__); \
        fprintf(stderr, "\n");          \
    } while (0);

#define ATOMIC_ADD(addr, delta) std::atomic_fetch_add(addr, delta)

#endif