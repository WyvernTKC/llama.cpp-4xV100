#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <cstdio>

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    llama_file(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file(FILE * file);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    // positional read: does not use or disturb the shared file position, so several threads may
    // call it concurrently on the same llama_file. leaves the file position undefined - always
    // seek() before going back to the sequential read_raw() path.
    //
    // concurrent calls on ONE llama_file are safe but not parallel: on Windows the I/O manager
    // serialises every request on a synchronous file object. use reopen() to give each thread its
    // own handle when the point is to overlap the reads.
    void read_raw_at(void * ptr, size_t len, size_t offset) const;

    // open an independent handle on the same file, or nullptr if this llama_file was built from a
    // FILE * and has no path to reopen
    std::unique_ptr<llama_file> reopen() const;

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    // list of [first, last) byte ranges within a file
    using ranges = std::vector<std::pair<size_t, size_t>>;

    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false,
               const ranges & lazy_ranges = {});
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t llama_path_max();

// ask the OS to make several possibly unaligned ranges resident. the ranges are page-aligned,
// sorted and merged first. this is only a hint, demand paging stays the fallback
void llama_prefetch_ranges(const void * const * addrs, const size_t * sizes, size_t n);
