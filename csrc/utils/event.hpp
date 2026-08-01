#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <memory>

#include <deep_ep/common/exception.cuh>

namespace deep_ep {

struct EventHandle {
    std::shared_ptr<torch::Event> event;
    // Keep communication inputs and temporaries alive through the producer
    // event.  Retention alone is sufficient for tensors allocated on the
    // compute stream: the wait orders their eventual release after comm.
    std::vector<torch::Tensor> tensors_to_retain;
    // Stable storage aliases for communication-owned tensors that may have
    // later compute-stream readers. Keeping Storage instead of Tensor makes
    // the allocator publication immune to callers rebinding ``tensor.data``.
    std::vector<c10::Storage> storages_to_record;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(at::cuda::getCurrentCUDAStream());
    }

    explicit EventHandle(const at::cuda::CUDAStream& stream) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void wait_and_publish(const at::cuda::CUDAStream& stream) const {
        stream.unwrap().wait(*event);
        for (const auto& storage: storages_to_record)
            c10::cuda::CUDACachingAllocator::recordStream(storage.data_ptr(), stream);
    }

    // Compatibility entry point for Python callers. This is an action, not an
    // observation: it both waits and publishes the actual allocator consumer.
    void current_stream_wait() const {
        wait_and_publish(at::cuda::getCurrentCUDAStream());
    }
};

static torch::Event create_event(const at::cuda::CUDAStream& s) {
    auto event = torch::Event(torch::kCUDA);
    event.record(s);
    return event;
}

static void stream_wait(const at::cuda::CUDAStream& s_0, const at::cuda::CUDAStream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

static void stream_wait(const at::cuda::CUDAStream& s, const EventHandle& event) {
    event.wait_and_publish(s);
}

}  // namespace deep_ep
