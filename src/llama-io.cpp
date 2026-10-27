#include "llama-io.h"

#include <stdexcept>
#include <vector>

void llama_io_write_i::write_tensor_2d(size_t /* size */, const std::vector<llama_io_tensor_2d> & /* parts */) {
    throw std::runtime_error("this state writer has no 2-D tensor copies");
}

void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_read_i::read_tensor_2d(size_t /* size */, const std::vector<llama_io_tensor_2d> & /* parts */) {
    throw std::runtime_error("this state reader has no 2-D tensor copies");
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read(&str_size, sizeof(str_size));

    std::vector<char> buf(str_size);
    read(buf.data(), str_size);

    str.assign(buf.data(), str_size);
}
