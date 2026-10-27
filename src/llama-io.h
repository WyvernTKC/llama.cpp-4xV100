#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;

// a 2-D copy between a tensor and a block of state data: n_rows chunks of size bytes,
//   at offset + i*stride in the tensor and at data_offset + i*data_stride in the block
struct llama_io_tensor_2d {
    ggml_tensor * tensor;
    size_t offset;
    size_t size;
    size_t n_rows;
    size_t stride;
    size_t data_offset;
    size_t data_stride;
};

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // write a block of size bytes that the 2-D copies fill; only if has_tensor_2d()
    virtual bool has_tensor_2d() const { return false; }
    virtual void write_tensor_2d(size_t size, const std::vector<llama_io_tensor_2d> & parts);

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // read a block of size bytes that the 2-D copies take from; only if has_tensor_2d()
    virtual bool has_tensor_2d() const { return false; }
    virtual void read_tensor_2d(size_t size, const std::vector<llama_io_tensor_2d> & parts);

    // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
    virtual void discard() {}

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
