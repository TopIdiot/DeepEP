#include "transport_abi.h"

#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/python.h>
#include <torch/version.h>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "buffer.hpp"

// The same overlay also remains buildable for the older non-namespaced
// ElasticBuffer used by tools/docker_v2/Dockerfile. Only the pinned
// deep_ep_ring source tree has the five-field ownership interface, so only
// that provider may advertise the capability bit.
#if __has_include("../../deep_ep_ring/__init__.py")
#define MMQ_DEEPEP_RING_OWNERSHIP_PROVIDER 1
#else
#define MMQ_DEEPEP_RING_OWNERSHIP_PROVIDER 0
#endif

namespace py = pybind11;
using deep_ep::EventHandle;
using deep_ep::elastic::ElasticBuffer;

struct DeepEpTransportRuntimeV1 {
  std::atomic<uint32_t> refs{1};
  PyObject *owner = nullptr;
  ElasticBuffer *buffer = nullptr;
};

struct DeepEpTransportEventV1 {
  std::atomic<uint32_t> refs{1};
  EventHandle event;

  explicit DeepEpTransportEventV1(const EventHandle &event) : event(event) {}
};

struct DeepEpTransportHandleV1 {
  std::atomic<uint32_t> refs{1};
  DeepEpTransportRuntimeV1 *runtime;
  bool do_expand;
  int num_experts;
  int expert_alignment;
  int num_max_tokens_per_rank;
  int num_sms;
  torch::Tensor topk_idx;
  int num_recv_tokens;
  int num_expanded_tokens;
  std::vector<int> num_recv_tokens_per_expert_list;
  torch::Tensor psum_num_recv_tokens_per_scaleup_rank;
  torch::Tensor psum_num_recv_tokens_per_expert;
  torch::Tensor num_unaligned_recv_tokens_per_expert;
  torch::Tensor recv_src_metadata;
  torch::Tensor dst_buffer_slot_idx;
  std::optional<torch::Tensor> token_metadata_at_forward;
  std::optional<torch::Tensor> channel_linked_list;

  DeepEpTransportHandleV1(
      DeepEpTransportRuntimeV1 *runtime, bool do_expand, int num_experts,
      int expert_alignment, int num_max_tokens_per_rank, int num_sms,
      torch::Tensor topk_idx, int num_recv_tokens, int num_expanded_tokens,
      std::vector<int> num_recv_tokens_per_expert_list,
      torch::Tensor psum_num_recv_tokens_per_scaleup_rank,
      torch::Tensor psum_num_recv_tokens_per_expert,
      torch::Tensor num_unaligned_recv_tokens_per_expert,
      torch::Tensor recv_src_metadata, torch::Tensor dst_buffer_slot_idx,
      std::optional<torch::Tensor> token_metadata_at_forward,
      std::optional<torch::Tensor> channel_linked_list);

  ~DeepEpTransportHandleV1();
};

namespace {

thread_local std::string g_last_error;

class TransportFailure : public std::runtime_error {
public:
  int32_t status;

  TransportFailure(int32_t status, const std::string &message)
      : std::runtime_error(message), status(status) {}
};

[[noreturn]] void fail(int32_t status, const std::string &message) {
  throw TransportFailure(status, message);
}

template <typename T> void validate_struct(const T *value, const char *name) {
  if (value == nullptr)
    fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
         std::string(name) + " is null");
  if (value->abi_version != DEEPEP_TRANSPORT_ABI_VERSION)
    fail(DEEPEP_TRANSPORT_STATUS_ABI_MISMATCH,
         std::string(name) + " has an incompatible ABI version");
  if (value->struct_size < sizeof(T))
    fail(DEEPEP_TRANSPORT_STATUS_ABI_MISMATCH,
         std::string(name) + " is truncated");
}

template <typename T>
void validate_struct_prefix(const T *value, size_t prefix_size,
                            const char *name) {
  if (value == nullptr)
    fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
         std::string(name) + " is null");
  if (value->abi_version != DEEPEP_TRANSPORT_ABI_VERSION)
    fail(DEEPEP_TRANSPORT_STATUS_ABI_MISMATCH,
         std::string(name) + " has an incompatible ABI version");
  if (value->struct_size < prefix_size)
    fail(DEEPEP_TRANSPORT_STATUS_ABI_MISMATCH,
         std::string(name) + " is truncated");
}

// Enforce the no-implicit-trailing-padding rule documented in transport_abi.h:
// struct_size (= sizeof) must equal the end of the last field, or the tail
// checks below would decide presence from a consumer's padding bytes.
static_assert(sizeof(DeepEpTransportDispatchRequestV1) ==
                  offsetof(DeepEpTransportDispatchRequestV1,
                           caller_managed_dispatch_reserved) +
                      sizeof(uint32_t),
              "DeepEpTransportDispatchRequestV1 has implicit trailing padding");
static_assert(sizeof(DeepEpTransportCombineRequestV1) ==
                  offsetof(DeepEpTransportCombineRequestV1,
                           caller_managed_combine_output_compute_owner) +
                      sizeof(uint32_t),
              "DeepEpTransportCombineRequestV1 has implicit trailing padding");
static_assert(sizeof(DeepEpTransportDispatchResponseV1) ==
                  offsetof(DeepEpTransportDispatchResponseV1,
                           channel_linked_list_out) +
                      sizeof(DeepEpTransportTensorSlotV1),
              "DeepEpTransportDispatchResponseV1 has implicit trailing "
              "padding");

template <typename T>
bool struct_has_field(const T *value, size_t offset, size_t field_size) {
  return value->struct_size >= offset + field_size;
}

template <typename T>
bool optional_u32_flag(const T *value, size_t offset,
                       uint32_t T::*field, const char *name) {
  if (!struct_has_field(value, offset, sizeof(value->*field)))
    return false;
  const uint32_t raw = value->*field;
  if (raw > 1)
    fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
         std::string(name) + " must be 0 or 1");
  return raw != 0;
}

template <typename Callback> int32_t guarded(Callback &&callback) {
  try {
    callback();
    g_last_error.clear();
    return DEEPEP_TRANSPORT_STATUS_OK;
  } catch (const TransportFailure &error) {
    g_last_error = error.what();
    return error.status;
  } catch (const c10::Error &error) {
    g_last_error = error.what();
  } catch (const py::error_already_set &error) {
    g_last_error = error.what();
  } catch (const std::exception &error) {
    g_last_error = error.what();
  } catch (...) {
    g_last_error = "unknown DeepEP transport error";
  }
  return DEEPEP_TRANSPORT_STATUS_ERROR;
}

const torch::Tensor &required_tensor(const void *value, const char *name) {
  if (value == nullptr)
    fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
         std::string(name) + " is null");
  return *static_cast<const torch::Tensor *>(value);
}

std::optional<torch::Tensor> optional_tensor(const void *value) {
  if (value == nullptr)
    return std::nullopt;
  return *static_cast<const torch::Tensor *>(value);
}

void store_tensor(void *slot, const torch::Tensor &value, const char *name) {
  if (slot == nullptr)
    fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
         std::string(name) + " output slot is null");
  *static_cast<torch::Tensor *>(slot) = value;
}

void store_optional_tensor(void *slot,
                           const std::optional<torch::Tensor> &value) {
  if (slot != nullptr)
    *static_cast<torch::Tensor *>(slot) = value.value_or(torch::Tensor());
}

std::optional<EventHandle> optional_event(DeepEpTransportEventV1 *event) {
  if (event == nullptr)
    return std::nullopt;
  return event->event;
}

int32_t runtime_retain_impl(DeepEpTransportRuntimeV1 *runtime) {
  if (runtime == nullptr)
    fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT, "runtime is null");
  runtime->refs.fetch_add(1, std::memory_order_relaxed);
  return DEEPEP_TRANSPORT_STATUS_OK;
}

int32_t runtime_release_impl(DeepEpTransportRuntimeV1 *runtime) {
  if (runtime == nullptr)
    fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT, "runtime is null");
  if (runtime->refs.fetch_sub(1, std::memory_order_acq_rel) != 1)
    return DEEPEP_TRANSPORT_STATUS_OK;

  if (runtime->owner != nullptr && Py_IsInitialized()) {
    const auto gil_state = PyGILState_Ensure();
    try {
      Py_DECREF(runtime->owner);
      runtime->owner = nullptr;
    } catch (...) {
      PyGILState_Release(gil_state);
      delete runtime;
      throw;
    }
    PyGILState_Release(gil_state);
  }
  delete runtime;
  return DEEPEP_TRANSPORT_STATUS_OK;
}

int32_t handle_retain_impl(DeepEpTransportHandleV1 *handle) {
  if (handle == nullptr)
    fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT, "handle is null");
  handle->refs.fetch_add(1, std::memory_order_relaxed);
  return DEEPEP_TRANSPORT_STATUS_OK;
}

int32_t handle_release_impl(DeepEpTransportHandleV1 *handle) {
  if (handle == nullptr)
    fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT, "handle is null");
  if (handle->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    auto *runtime = handle->runtime;
    handle->runtime = nullptr;
    delete handle;
    runtime_release_impl(runtime);
  }
  return DEEPEP_TRANSPORT_STATUS_OK;
}

int32_t event_retain_impl(DeepEpTransportEventV1 *event) {
  if (event == nullptr)
    fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT, "event is null");
  event->refs.fetch_add(1, std::memory_order_relaxed);
  return DEEPEP_TRANSPORT_STATUS_OK;
}

int32_t event_release_impl(DeepEpTransportEventV1 *event) {
  if (event == nullptr)
    fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT, "event is null");
  if (event->refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
    delete event;
  return DEEPEP_TRANSPORT_STATUS_OK;
}

template <typename LifetimeFunction, typename Value>
int32_t guarded_lifetime(LifetimeFunction function, Value *value) {
  return guarded([&] { function(value); });
}

int32_t runtime_retain_v1(DeepEpTransportRuntimeV1 *runtime) {
  return guarded_lifetime(runtime_retain_impl, runtime);
}

int32_t runtime_release_v1(DeepEpTransportRuntimeV1 *runtime) {
  return guarded_lifetime(runtime_release_impl, runtime);
}

int32_t handle_retain_v1(DeepEpTransportHandleV1 *handle) {
  return guarded_lifetime(handle_retain_impl, handle);
}

int32_t handle_release_v1(DeepEpTransportHandleV1 *handle) {
  return guarded_lifetime(handle_release_impl, handle);
}

int32_t event_retain_v1(DeepEpTransportEventV1 *event) {
  return guarded_lifetime(event_retain_impl, event);
}

int32_t event_release_v1(DeepEpTransportEventV1 *event) {
  return guarded_lifetime(event_release_impl, event);
}

int32_t runtime_from_pyobject_v1(void *py_object,
                                 DeepEpTransportRuntimeV1 **runtime_out) {
  return guarded([&] {
    if (py_object == nullptr || runtime_out == nullptr)
      fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
           "runtime_from_pyobject received a null argument");
    *runtime_out = nullptr;

    py::gil_scoped_acquire acquire;
    auto *owner = static_cast<PyObject *>(py_object);
    auto *buffer = py::cast<ElasticBuffer *>(py::handle(owner));
    if (buffer == nullptr)
      fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
           "PyObject is not a DeepEP ElasticBuffer");

    auto runtime = std::make_unique<DeepEpTransportRuntimeV1>();
    Py_INCREF(owner);
    runtime->owner = owner;
    runtime->buffer = buffer;
    *runtime_out = runtime.release();
  });
}

int32_t event_from_pyobject_v1(void *py_object,
                               DeepEpTransportEventV1 **event_out) {
  return guarded([&] {
    if (py_object == nullptr || event_out == nullptr)
      fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
           "event_from_pyobject received a null argument");
    *event_out = nullptr;

    py::gil_scoped_acquire acquire;
    auto event =
        py::cast<EventHandle>(py::handle(static_cast<PyObject *>(py_object)));
    *event_out = new DeepEpTransportEventV1(event);
  });
}

int32_t dispatch_v1_impl(const DeepEpTransportDispatchRequestV1 *request,
                         DeepEpTransportDispatchResponseV1 *response,
                         bool with_handle) {
  return guarded([&] {
    validate_struct_prefix(
        request,
        offsetof(DeepEpTransportDispatchRequestV1,
                 caller_managed_dispatch_input_lifetime),
        "dispatch request");
    validate_struct_prefix(
        response,
        offsetof(DeepEpTransportDispatchResponseV1, handle_topk_idx_out),
        "dispatch response");
    response->handle = nullptr;
    response->event = nullptr;

    auto *runtime = request->runtime;
    if (runtime == nullptr || runtime->buffer == nullptr)
      fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
           "dispatch runtime is null");
    if (response->recv_x_out == nullptr ||
        response->psum_num_recv_tokens_per_scaleup_rank_out == nullptr ||
        response->psum_num_recv_tokens_per_expert_out == nullptr ||
        response->recv_src_metadata_out == nullptr)
      fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
           "dispatch required output slot is null");
    auto *cached_handle = request->cached_handle;
    if (with_handle != (cached_handle != nullptr))
      fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
           "dispatch cached-handle mode does not match entrypoint");
    if (cached_handle != nullptr) {
      if (cached_handle->runtime != runtime)
        fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
             "dispatch handle belongs to another runtime");
      if (cached_handle->do_expand)
        fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
             "cached expanded dispatch is unsupported");
    }

    const auto &x = required_tensor(request->x, "dispatch x");
    const torch::Tensor *topk_idx =
        cached_handle == nullptr
            ? &required_tensor(request->topk_idx, "dispatch topk_idx")
            : &cached_handle->topk_idx;
    const auto sf = optional_tensor(request->sf);
    const auto topk_weights = optional_tensor(request->topk_weights);
    const auto cumulative_stats =
        optional_tensor(request->cumulative_local_expert_recv_stats);
    const bool caller_managed_dispatch_input_lifetime = optional_u32_flag(
        request,
        offsetof(DeepEpTransportDispatchRequestV1,
                 caller_managed_dispatch_input_lifetime),
        &DeepEpTransportDispatchRequestV1::caller_managed_dispatch_input_lifetime,
        "caller_managed_dispatch_input_lifetime");
    const bool caller_managed_dispatch_recv_lifetime = optional_u32_flag(
        request,
        offsetof(DeepEpTransportDispatchRequestV1,
                 caller_managed_dispatch_recv_lifetime),
        &DeepEpTransportDispatchRequestV1::caller_managed_dispatch_recv_lifetime,
        "caller_managed_dispatch_recv_lifetime");
    const bool caller_managed_dispatch_recv_compute_owner = optional_u32_flag(
        request,
        offsetof(DeepEpTransportDispatchRequestV1,
                 caller_managed_dispatch_recv_compute_owner),
        &DeepEpTransportDispatchRequestV1::caller_managed_dispatch_recv_compute_owner,
        "caller_managed_dispatch_recv_compute_owner");

    const auto cached_num_recv_tokens =
        cached_handle == nullptr
            ? std::optional<int>()
            : std::optional<int>(cached_handle->num_recv_tokens);
    const auto cached_num_expanded_tokens =
        cached_handle == nullptr
            ? std::optional<int>()
            : std::optional<int>(cached_handle->num_expanded_tokens);
    const auto cached_num_recv_tokens_per_expert_list =
        cached_handle == nullptr
            ? std::optional<std::vector<int>>()
            : std::optional<std::vector<int>>(
                  cached_handle->num_recv_tokens_per_expert_list);
    const auto cached_psum_scaleup =
        cached_handle == nullptr
            ? std::optional<torch::Tensor>()
            : std::optional<torch::Tensor>(
                  cached_handle->psum_num_recv_tokens_per_scaleup_rank);
    const auto cached_psum_expert =
        cached_handle == nullptr
            ? std::optional<torch::Tensor>()
            : std::optional<torch::Tensor>(
                  cached_handle->psum_num_recv_tokens_per_expert);
    const auto cached_num_unaligned =
        cached_handle == nullptr
            ? std::optional<torch::Tensor>()
            : std::optional<torch::Tensor>(
                  cached_handle->num_unaligned_recv_tokens_per_expert);
    const auto cached_dst_slot =
        cached_handle == nullptr
            ? std::optional<torch::Tensor>()
            : std::optional<torch::Tensor>(cached_handle->dst_buffer_slot_idx);
    const auto cached_token_metadata =
        cached_handle == nullptr ? std::optional<torch::Tensor>()
                                 : cached_handle->token_metadata_at_forward;
    const auto cached_recv_src_metadata =
        cached_handle == nullptr
            ? std::optional<torch::Tensor>()
            : std::optional<torch::Tensor>(cached_handle->recv_src_metadata);
    const auto cached_channel_list = cached_handle == nullptr
                                         ? std::optional<torch::Tensor>()
                                         : cached_handle->channel_linked_list;

    torch::Tensor recv_x;
    std::optional<torch::Tensor> recv_sf;
    std::optional<torch::Tensor> recv_topk_idx;
    std::optional<torch::Tensor> recv_topk_weights;
    std::optional<torch::Tensor> copied_topk_idx;
    int num_recv_tokens = 0;
    int num_expanded_tokens = 0;
    std::vector<int> num_recv_tokens_per_expert_list;
    torch::Tensor psum_scaleup;
    torch::Tensor psum_expert;
    torch::Tensor num_unaligned_recv_tokens_per_expert;
    torch::Tensor recv_src_metadata;
    torch::Tensor dst_buffer_slot_idx;
    std::optional<torch::Tensor> token_metadata_at_forward;
    std::optional<torch::Tensor> channel_linked_list;
    std::optional<EventHandle> event;

#if MMQ_DEEPEP_RING_OWNERSHIP_PROVIDER
    if (request->previous_event_before_epilogue != nullptr)
      fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
           "deep_ep_ring dispatch does not support previous_event_before_epilogue");
    auto result = runtime->buffer->dispatch(
        x, sf, *topk_idx, topk_weights, cumulative_stats,
        cached_num_recv_tokens, cached_num_expanded_tokens,
        cached_num_recv_tokens_per_expert_list, cached_psum_scaleup,
        cached_psum_expert, cached_num_unaligned, cached_dst_slot,
        cached_token_metadata, cached_recv_src_metadata, cached_channel_list,
        cached_handle == nullptr ? request->num_max_tokens_per_rank
                                 : cached_handle->num_max_tokens_per_rank,
        cached_handle == nullptr ? request->num_experts
                                 : cached_handle->num_experts,
        cached_handle == nullptr ? request->expert_alignment
                                 : cached_handle->expert_alignment,
        cached_handle == nullptr ? request->num_sms : cached_handle->num_sms,
        request->num_qps, optional_event(request->previous_event),
        request->async_with_compute_stream != 0,
        request->allocate_on_comm_stream != 0,
        cached_handle == nullptr && request->do_handle_copy != 0,
        cached_handle == nullptr && request->do_cpu_sync != 0,
        cached_handle == nullptr && request->do_expand != 0,
        false, /* do_zero_padding */
        request->use_tma_aligned_col_major_sf != 0,
        caller_managed_dispatch_input_lifetime,
        caller_managed_dispatch_recv_lifetime,
        caller_managed_dispatch_recv_compute_owner);

    recv_x = std::get<0>(result);
    recv_sf = std::get<1>(result);
    recv_topk_idx = std::get<2>(result);
    recv_topk_weights = std::get<3>(result);
    copied_topk_idx = std::get<4>(result);
    num_recv_tokens = std::get<5>(result);
    num_expanded_tokens = std::get<6>(result);
    num_recv_tokens_per_expert_list = std::move(std::get<7>(result));
    psum_scaleup = std::get<8>(result);
    psum_expert = std::get<9>(result);
    num_unaligned_recv_tokens_per_expert = std::get<10>(result);
    recv_src_metadata = std::get<11>(result);
    dst_buffer_slot_idx = std::get<12>(result);
    token_metadata_at_forward = std::get<13>(result);
    channel_linked_list = std::get<14>(result);
    event = std::get<15>(result);
#else
    if (caller_managed_dispatch_input_lifetime ||
        caller_managed_dispatch_recv_lifetime ||
        caller_managed_dispatch_recv_compute_owner)
      fail(DEEPEP_TRANSPORT_STATUS_ABI_MISMATCH,
           "provider lacks caller-managed ownership capability");
    auto result = runtime->buffer->dispatch(
        x, sf, *topk_idx, topk_weights, cumulative_stats,
        cached_num_recv_tokens, cached_num_recv_tokens_per_expert_list,
        cached_psum_scaleup, cached_psum_expert, cached_dst_slot,
        cached_token_metadata, cached_channel_list,
        cached_handle == nullptr ? request->num_max_tokens_per_rank
                                 : cached_handle->num_max_tokens_per_rank,
        cached_handle == nullptr ? request->num_experts
                                 : cached_handle->num_experts,
        cached_handle == nullptr ? request->expert_alignment
                                 : cached_handle->expert_alignment,
        cached_handle == nullptr ? request->num_sms : cached_handle->num_sms,
        request->num_qps, optional_event(request->previous_event),
        optional_event(request->previous_event_before_epilogue),
        request->async_with_compute_stream != 0,
        request->allocate_on_comm_stream != 0,
        cached_handle == nullptr && request->do_handle_copy != 0,
        cached_handle == nullptr && request->do_cpu_sync != 0,
        cached_handle == nullptr && request->do_expand != 0,
        request->use_tma_aligned_col_major_sf != 0);

    recv_x = std::get<0>(result);
    recv_sf = std::get<1>(result);
    recv_topk_idx = std::get<2>(result);
    recv_topk_weights = std::get<3>(result);
    copied_topk_idx = std::get<4>(result);
    num_recv_tokens_per_expert_list = std::move(std::get<5>(result));
    psum_scaleup = std::get<6>(result);
    psum_expert = std::get<7>(result);
    recv_src_metadata = std::get<8>(result);
    num_recv_tokens = static_cast<int>(recv_src_metadata.size(0));
    num_expanded_tokens = num_recv_tokens;
    dst_buffer_slot_idx = std::get<9>(result);
    token_metadata_at_forward = std::get<10>(result);
    channel_linked_list = std::get<11>(result);
    event = std::get<12>(result);
#endif

    store_tensor(response->recv_x_out, recv_x, "recv_x");
    store_optional_tensor(response->recv_sf_out, recv_sf);
    store_optional_tensor(response->recv_topk_idx_out, recv_topk_idx);
    store_optional_tensor(response->recv_topk_weights_out, recv_topk_weights);
    store_tensor(response->psum_num_recv_tokens_per_scaleup_rank_out,
                 psum_scaleup, "psum_num_recv_tokens_per_scaleup_rank");
    store_tensor(response->psum_num_recv_tokens_per_expert_out, psum_expert,
                 "psum_num_recv_tokens_per_expert");
    store_tensor(response->recv_src_metadata_out, recv_src_metadata,
                 "recv_src_metadata");
    if (struct_has_field(
            response,
            offsetof(DeepEpTransportDispatchResponseV1, handle_topk_idx_out),
            sizeof(response->handle_topk_idx_out)))
      store_tensor(response->handle_topk_idx_out,
                   copied_topk_idx.value_or(*topk_idx), "handle_topk_idx");
    if (struct_has_field(response,
                         offsetof(DeepEpTransportDispatchResponseV1,
                                  dst_buffer_slot_idx_out),
                         sizeof(response->dst_buffer_slot_idx_out)))
      store_tensor(response->dst_buffer_slot_idx_out, dst_buffer_slot_idx,
                   "dst_buffer_slot_idx");
    if (struct_has_field(response,
                         offsetof(DeepEpTransportDispatchResponseV1,
                                  token_metadata_at_forward_out),
                         sizeof(response->token_metadata_at_forward_out)))
      store_optional_tensor(response->token_metadata_at_forward_out,
                            token_metadata_at_forward);
    if (struct_has_field(response,
                         offsetof(DeepEpTransportDispatchResponseV1,
                                  channel_linked_list_out),
                         sizeof(response->channel_linked_list_out)))
      store_optional_tensor(response->channel_linked_list_out,
                            channel_linked_list);

    std::unique_ptr<DeepEpTransportHandleV1> new_handle;
    if (cached_handle == nullptr) {
      new_handle = std::make_unique<DeepEpTransportHandleV1>(
          runtime, request->do_expand != 0, request->num_experts,
          request->expert_alignment, request->num_max_tokens_per_rank,
          request->num_sms, copied_topk_idx.value_or(*topk_idx),
          num_recv_tokens, num_expanded_tokens,
          std::move(num_recv_tokens_per_expert_list), psum_scaleup, psum_expert,
          num_unaligned_recv_tokens_per_expert, recv_src_metadata,
          dst_buffer_slot_idx, token_metadata_at_forward, channel_linked_list);
    }
    std::unique_ptr<DeepEpTransportEventV1> new_event;
    if (event.has_value())
      new_event = std::make_unique<DeepEpTransportEventV1>(event.value());

    if (cached_handle == nullptr) {
      response->handle = new_handle.release();
    } else {
      handle_retain_impl(cached_handle);
      response->handle = cached_handle;
    }
    response->event = new_event.release();
  });
}

int32_t dispatch_v1(const DeepEpTransportDispatchRequestV1 *request,
                    DeepEpTransportDispatchResponseV1 *response) {
  return dispatch_v1_impl(request, response, false);
}

int32_t dispatch_with_handle_v1(const DeepEpTransportDispatchRequestV1 *request,
                                DeepEpTransportDispatchResponseV1 *response) {
  return dispatch_v1_impl(request, response, true);
}

int32_t combine_v1(const DeepEpTransportCombineRequestV1 *request,
                   DeepEpTransportCombineResponseV1 *response) {
  return guarded([&] {
    validate_struct_prefix(
        request,
        offsetof(DeepEpTransportCombineRequestV1,
                 caller_managed_combine_input_lifetime),
        "combine request");
    validate_struct(response, "combine response");
    response->event = nullptr;

    auto *runtime = request->runtime;
    auto *handle = request->handle;
    if (runtime == nullptr || runtime->buffer == nullptr || handle == nullptr)
      fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
           "combine runtime or handle is null");
    if (response->combined_x_out == nullptr)
      fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
           "combine required output slot is null");
    if (handle->runtime != runtime)
      fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
           "combine handle belongs to another runtime");
    const bool caller_managed_combine_input_lifetime = optional_u32_flag(
        request,
        offsetof(DeepEpTransportCombineRequestV1,
                 caller_managed_combine_input_lifetime),
        &DeepEpTransportCombineRequestV1::caller_managed_combine_input_lifetime,
        "caller_managed_combine_input_lifetime");
    const bool caller_managed_combine_output_compute_owner = optional_u32_flag(
        request,
        offsetof(DeepEpTransportCombineRequestV1,
                 caller_managed_combine_output_compute_owner),
        &DeepEpTransportCombineRequestV1::caller_managed_combine_output_compute_owner,
        "caller_managed_combine_output_compute_owner");

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<EventHandle>> result;
#if MMQ_DEEPEP_RING_OWNERSHIP_PROVIDER
    if (request->previous_event_before_epilogue != nullptr)
      fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT,
           "deep_ep_ring combine does not support previous_event_before_epilogue");
    result = runtime->buffer->combine(
        required_tensor(request->x, "combine x"),
        optional_tensor(request->topk_weights),
        optional_tensor(request->bias_0), optional_tensor(request->bias_1),
        handle->recv_src_metadata, handle->topk_idx,
        handle->psum_num_recv_tokens_per_scaleup_rank,
        handle->token_metadata_at_forward, handle->channel_linked_list,
        handle->num_experts, handle->num_max_tokens_per_rank,
        request->num_sms == 0 ? handle->num_sms : request->num_sms,
        request->num_qps, optional_event(request->previous_event),
        request->async_with_compute_stream != 0,
        request->allocate_on_comm_stream != 0, handle->do_expand,
        caller_managed_combine_input_lifetime,
        caller_managed_combine_output_compute_owner);
#else
    if (caller_managed_combine_input_lifetime ||
        caller_managed_combine_output_compute_owner)
      fail(DEEPEP_TRANSPORT_STATUS_ABI_MISMATCH,
           "provider lacks caller-managed ownership capability");
    result = runtime->buffer->combine(
        required_tensor(request->x, "combine x"),
        optional_tensor(request->topk_weights),
        optional_tensor(request->bias_0), optional_tensor(request->bias_1),
        handle->recv_src_metadata, handle->topk_idx,
        handle->psum_num_recv_tokens_per_scaleup_rank,
        handle->token_metadata_at_forward, handle->channel_linked_list,
        handle->num_experts, handle->num_max_tokens_per_rank,
        request->num_sms == 0 ? handle->num_sms : request->num_sms,
        request->num_qps, optional_event(request->previous_event),
        optional_event(request->previous_event_before_epilogue),
        request->async_with_compute_stream != 0,
        request->allocate_on_comm_stream != 0, handle->do_expand);
#endif

    auto &combined_x = std::get<0>(result);
    auto &combined_topk_weights = std::get<1>(result);
    auto &event = std::get<2>(result);
    store_tensor(response->combined_x_out, combined_x, "combined_x");
    store_optional_tensor(response->combined_topk_weights_out,
                          combined_topk_weights);

    if (event.has_value())
      response->event = new DeepEpTransportEventV1(event.value());
  });
}

int32_t event_wait_v1(DeepEpTransportEventV1 *event) {
  return guarded([&] {
    if (event == nullptr)
      fail(DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT, "event is null");
    event->event.current_stream_wait();
  });
}

const char *last_error_v1() { return g_last_error.c_str(); }

const DeepEpTransportApiV1 kTransportApiV1 = {
    sizeof(DeepEpTransportApiV1),
    DEEPEP_TRANSPORT_ABI_VERSION,
    DEEPEP_TRANSPORT_CAP_DISPATCH | DEEPEP_TRANSPORT_CAP_COMBINE |
        DEEPEP_TRANSPORT_CAP_DISPATCH_WITH_HANDLE |
        DEEPEP_TRANSPORT_CAP_HANDLE_LIFETIME |
        DEEPEP_TRANSPORT_CAP_EVENT_LIFETIME |
        DEEPEP_TRANSPORT_CAP_RUNTIME_LIFETIME |
        DEEPEP_TRANSPORT_CAP_EVENT_FROM_PYOBJECT
#if MMQ_DEEPEP_RING_OWNERSHIP_PROVIDER
        | DEEPEP_TRANSPORT_CAP_CALLER_MANAGED_OWNERSHIP
#endif
    ,
    runtime_from_pyobject_v1,
    dispatch_v1,
    combine_v1,
    dispatch_with_handle_v1,
    handle_retain_v1,
    handle_release_v1,
    event_retain_v1,
    event_release_v1,
    event_wait_v1,
    runtime_retain_v1,
    runtime_release_v1,
    last_error_v1,
#ifdef _GLIBCXX_USE_CXX11_ABI
    static_cast<uint32_t>(_GLIBCXX_USE_CXX11_ABI),
#else
    0,
#endif
    static_cast<uint32_t>(sizeof(torch::Tensor)),
    TORCH_VERSION,
    event_from_pyobject_v1,
};

} // namespace

DeepEpTransportHandleV1::DeepEpTransportHandleV1(
    DeepEpTransportRuntimeV1 *runtime, bool do_expand, int num_experts,
    int expert_alignment, int num_max_tokens_per_rank, int num_sms,
    torch::Tensor topk_idx, int num_recv_tokens, int num_expanded_tokens,
    std::vector<int> num_recv_tokens_per_expert_list,
    torch::Tensor psum_num_recv_tokens_per_scaleup_rank,
    torch::Tensor psum_num_recv_tokens_per_expert,
    torch::Tensor num_unaligned_recv_tokens_per_expert,
    torch::Tensor recv_src_metadata, torch::Tensor dst_buffer_slot_idx,
    std::optional<torch::Tensor> token_metadata_at_forward,
    std::optional<torch::Tensor> channel_linked_list)
    : runtime(runtime), do_expand(do_expand), num_experts(num_experts),
      expert_alignment(expert_alignment),
      num_max_tokens_per_rank(num_max_tokens_per_rank), num_sms(num_sms),
      topk_idx(std::move(topk_idx)), num_recv_tokens(num_recv_tokens),
      num_expanded_tokens(num_expanded_tokens),
      num_recv_tokens_per_expert_list(
          std::move(num_recv_tokens_per_expert_list)),
      psum_num_recv_tokens_per_scaleup_rank(
          std::move(psum_num_recv_tokens_per_scaleup_rank)),
      psum_num_recv_tokens_per_expert(
          std::move(psum_num_recv_tokens_per_expert)),
      num_unaligned_recv_tokens_per_expert(
          std::move(num_unaligned_recv_tokens_per_expert)),
      recv_src_metadata(std::move(recv_src_metadata)),
      dst_buffer_slot_idx(std::move(dst_buffer_slot_idx)),
      token_metadata_at_forward(std::move(token_metadata_at_forward)),
      channel_linked_list(std::move(channel_linked_list)) {
  runtime_retain_impl(runtime);
}

DeepEpTransportHandleV1::~DeepEpTransportHandleV1() {
  if (runtime == nullptr)
    return;
  try {
    runtime_release_impl(runtime);
  } catch (...) {
    // Destructors reached during guarded error cleanup must never terminate.
  }
}

extern "C" DEEPEP_TRANSPORT_EXPORT const DeepEpTransportApiV1 *
deep_ep_transport_get_api_v1(void) {
  return &kTransportApiV1;
}
