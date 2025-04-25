#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <atomic>
#include <iostream>
#include <time.h>
#include <cstdio>
#define CUDA_CHECK(stmt)                                  \
do {                                                      \
    cudaError_t result = (stmt);                          \
    if (cudaSuccess != result) {                          \
        fprintf(stderr, "[%s:%d] CUDA failed with %s \n", \
         __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(-1);                                         \
    }                                                     \
} while (0)
namespace py = pybind11;
struct UniqueId {
    nvshmemx_uniqueid_t id;
    py::bytes UniqueId_to_bytes() {
        return py::bytes(reinterpret_cast<const char*>(&this->id), sizeof(nvshmemx_uniqueid_t));
    }
    static UniqueId bytes_to_UniqueId(const py::bytes& data) {
        UniqueId uid;
        const std::string& buffer = data;
        if (buffer.size() != sizeof(nvshmemx_uniqueid_t)) {
            throw std::runtime_error("Invalid bytes len for UniqueId");
        }
        memcpy(&uid.id, buffer.data(), sizeof(nvshmemx_uniqueid_t));
        return uid;
    }
    static UniqueId get_UniqueId() {
        UniqueId uid;
        nvshmemx_get_uniqueid(&uid.id);
        return uid;
    }
};

class c_buffer {
private:
    int local_rank, remote_rank;
    char* shared_mem;
    int insert_idx, pop_idx;
    size_t capacity, tensor_size, element_size;
public:
    c_buffer(int local_rank, int remote_rank, int world_size, UniqueId &uid) {
        this->local_rank = local_rank;
        this->remote_rank = remote_rank;
        const std::string& buffer = uid.UniqueId_to_bytes();
        std::cout << "\nc_buffer.cu: rank=" << local_rank << " uid=" << buffer << std::endl;

        nvshmemx_init_attr_t attr;
        nvshmemx_set_attr_uniqueid_args(local_rank, world_size, &uid.id, &attr);
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
        nvshmem_barrier_all();
        shared_mem = NULL;
        printf("\nc_buffer: rank = %d, initialized.\n", local_rank);
    }
    void allocate(size_t tensor_numel, size_t capacity,
                  const torch::ScalarType dtype,
                  const torch::Device& device) {
        
        this->element_size = torch::elementSize(dtype);
        this->tensor_size = tensor_numel * this->element_size;
        this->capacity = capacity;
        printf("\nc_buffer.cu: initialize start local_rank:%d remote_rank:%d device:%d element_size:%d\n", local_rank, remote_rank, device.index(), this->element_size);        

        assert(device.is_cuda());
        CUDA_CHECK(cudaSetDevice(device.index()));
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device.index());
        printf("\nc_buffer.cu: device uuid = %d busid = %d deviceid = %d\n", prop.uuid, prop.pciBusID, prop.pciDeviceID);
        int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        shared_mem = (char*)nvshmem_malloc(tensor_size * capacity);
        insert_idx = 0;
        pop_idx = capacity;
        printf("\n rank = %d, allocated\n", local_rank);
    }
    int is_allocated() {
        return shared_mem != NULL;
    }
    int sender_is_full() {
        return insert_idx == capacity;
    }
    int sender_any() {
        return insert_idx > 0;
    }
    int recver_is_empty() {
        return pop_idx == capacity;
    }
    void remote_put() {
        assert(sender_is_full());
        nvshmem_char_put(shared_mem, shared_mem, insert_idx * tensor_size, remote_rank);
        insert_idx = 0;
    }
    void remote_get() {
        assert(recver_is_empty());
        pop_idx = 0;
    }
    void insert_tensor(const torch::Tensor& tensor)
    {
        if (tensor.numel() * element_size != tensor_size)
            throw std::runtime_error("Invalid tensor for insert_tensor. tensor.numel() != tensor_size");
        cudaMemcpy(shared_mem + insert_idx * tensor_size,
                   tensor.data_ptr(),
                   tensor_size,
                   cudaMemcpyDeviceToDevice);
        insert_idx += 1;
    }
    void pop_tensor(const torch::Tensor& tensor)
    {
        if (tensor.numel() * element_size != tensor_size)
            throw std::runtime_error("Invalid tensor for pop_tensor. tensor.numel() != tensor_size");
        cudaMemcpy(tensor.data_ptr(),
                   shared_mem + pop_idx * tensor_size,
                   tensor_size,
                   cudaMemcpyDeviceToDevice);
        pop_idx += 1;
    }
    ~c_buffer() {
        nvshmem_free(shared_mem);
        nvshmem_finalize();
    }
};
PYBIND11_MODULE(c_buffer, m) {
    py::class_<UniqueId>(m, "UniqueId")
        .def(py::init<>())
        .def("to_bytes", &UniqueId::UniqueId_to_bytes)
        .def_static("from_bytes", &UniqueId::bytes_to_UniqueId)
        .def_static("get_uniqueid", &UniqueId::get_UniqueId);
    py::class_<c_buffer>(m, "c_buffer")
        .def(py::init<int, int, int, UniqueId&>())
        .def("allocate", &c_buffer::allocate)
        .def("is_allocated", &c_buffer::is_allocated)
        .def("insert_tensor", &c_buffer::insert_tensor)
        .def("pop_tensor", &c_buffer::pop_tensor)
        .def("sender_is_full", &c_buffer::sender_is_full)
        .def("sender_any", &c_buffer::sender_any)
        .def("recver_is_empty", &c_buffer::recver_is_empty)
        .def("remote_put", &c_buffer::remote_put)
        .def("remote_get", &c_buffer::remote_get);
}