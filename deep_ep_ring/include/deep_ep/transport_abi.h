#ifndef DEEPEP_TRANSPORT_ABI_H_
#define DEEPEP_TRANSPORT_ABI_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  DEEPEP_TRANSPORT_ABI_VERSION = 1,
};

enum DeepEpTransportCapabilityV1 {
  DEEPEP_TRANSPORT_CAP_DISPATCH = 1ull << 0,
  DEEPEP_TRANSPORT_CAP_COMBINE = 1ull << 1,
  DEEPEP_TRANSPORT_CAP_DISPATCH_WITH_HANDLE = 1ull << 2,
  DEEPEP_TRANSPORT_CAP_HANDLE_LIFETIME = 1ull << 3,
  DEEPEP_TRANSPORT_CAP_EVENT_LIFETIME = 1ull << 4,
  DEEPEP_TRANSPORT_CAP_RUNTIME_LIFETIME = 1ull << 5,
  DEEPEP_TRANSPORT_CAP_EVENT_FROM_PYOBJECT = 1ull << 6,
  /*
   * The provider reads the optional caller-managed ownership request tails
   * and forwards them without deriving or discarding any field.
   */
  DEEPEP_TRANSPORT_CAP_CALLER_MANAGED_OWNERSHIP = 1ull << 7,
};

enum DeepEpTransportStatusV1 {
  DEEPEP_TRANSPORT_STATUS_OK = 0,
  DEEPEP_TRANSPORT_STATUS_INVALID_ARGUMENT = 1,
  DEEPEP_TRANSPORT_STATUS_ABI_MISMATCH = 2,
  DEEPEP_TRANSPORT_STATUS_ERROR = 3,
};

typedef struct DeepEpTransportRuntimeV1 DeepEpTransportRuntimeV1;
typedef struct DeepEpTransportHandleV1 DeepEpTransportHandleV1;
typedef struct DeepEpTransportEventV1 DeepEpTransportEventV1;

/*
 * Tensor refs point to caller-owned at::Tensor objects and tensor slots point
 * to initialized caller-owned at::Tensor objects.  DeepEP assigns intrusive
 * tensor handles into output slots.
 *
 * The function table and opaque DeepEP handles have a versioned C ABI, but
 * this tensor bridge is deliberately a same-libtorch C++ ABI contract.  A
 * consumer MUST validate torch_version, torch_cxx11_abi, and
 * torch_tensor_size before dereferencing any function pointer.
 */
typedef const void *DeepEpTransportTensorRefV1;
typedef void *DeepEpTransportTensorSlotV1;

typedef struct DeepEpTransportDispatchRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeepEpTransportRuntimeV1 *runtime;
  DeepEpTransportHandleV1 *cached_handle;
  DeepEpTransportTensorRefV1 x;
  DeepEpTransportTensorRefV1 sf;
  DeepEpTransportTensorRefV1 topk_idx;
  DeepEpTransportTensorRefV1 topk_weights;
  DeepEpTransportTensorRefV1 cumulative_local_expert_recv_stats;
  int32_t num_max_tokens_per_rank;
  int32_t num_experts;
  int32_t expert_alignment;
  int32_t num_sms;
  int32_t num_qps;
  DeepEpTransportEventV1 *previous_event;
  DeepEpTransportEventV1 *previous_event_before_epilogue;
  uint32_t async_with_compute_stream;
  uint32_t allocate_on_comm_stream;
  uint32_t do_handle_copy;
  uint32_t do_cpu_sync;
  uint32_t do_expand;
  uint32_t use_tma_aligned_col_major_sf;
  /*
   * Optional v1 ownership tail. Providers must accept the original prefix
   * ending at use_tma_aligned_col_major_sf and treat every absent field as
   * false. Consumers may set any field only when the provider advertises
   * DEEPEP_TRANSPORT_CAP_CALLER_MANAGED_OWNERSHIP.
   *
   * Tail presence is derived from struct_size (= sizeof of the consumer's
   * revision), so every revision must end with no implicit trailing padding --
   * otherwise sizeof rounds up past the last real field and the provider reads
   * the next tail slot out of the consumer's padding. Append uint32_t tail
   * fields in pairs, or add an explicit reserved uint32_t. The provider's
   * static_asserts in tools/docker_v2/deepep_transport_abi.cpp enforce this.
   */
  uint32_t caller_managed_dispatch_input_lifetime;
  uint32_t caller_managed_dispatch_recv_lifetime;
  uint32_t caller_managed_dispatch_recv_compute_owner;
  /*
   * Explicit tail padding; see the struct_size note above. Consumers must
   * leave it zero.
   */
  uint32_t caller_managed_dispatch_reserved;
} DeepEpTransportDispatchRequestV1;

typedef struct DeepEpTransportDispatchResponseV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeepEpTransportTensorSlotV1 recv_x_out;
  DeepEpTransportTensorSlotV1 recv_sf_out;
  DeepEpTransportTensorSlotV1 recv_topk_idx_out;
  DeepEpTransportTensorSlotV1 recv_topk_weights_out;
  DeepEpTransportTensorSlotV1 psum_num_recv_tokens_per_scaleup_rank_out;
  DeepEpTransportTensorSlotV1 psum_num_recv_tokens_per_expert_out;
  DeepEpTransportTensorSlotV1 recv_src_metadata_out;
  DeepEpTransportHandleV1 *handle;
  DeepEpTransportEventV1 *event;
  /*
   * Optional v1 tail.  Providers must accept the original prefix ending at
   * event and only write each tail slot when struct_size includes that field.
   */
  DeepEpTransportTensorSlotV1 handle_topk_idx_out;
  DeepEpTransportTensorSlotV1 dst_buffer_slot_idx_out;
  DeepEpTransportTensorSlotV1 token_metadata_at_forward_out;
  DeepEpTransportTensorSlotV1 channel_linked_list_out;
} DeepEpTransportDispatchResponseV1;

typedef struct DeepEpTransportCombineRequestV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeepEpTransportRuntimeV1 *runtime;
  DeepEpTransportHandleV1 *handle;
  DeepEpTransportTensorRefV1 x;
  DeepEpTransportTensorRefV1 topk_weights;
  DeepEpTransportTensorRefV1 bias_0;
  DeepEpTransportTensorRefV1 bias_1;
  int32_t num_sms;
  int32_t num_qps;
  DeepEpTransportEventV1 *previous_event;
  DeepEpTransportEventV1 *previous_event_before_epilogue;
  uint32_t async_with_compute_stream;
  uint32_t allocate_on_comm_stream;
  /* Optional v1 ownership tail; absent fields mean false. */
  uint32_t caller_managed_combine_input_lifetime;
  uint32_t caller_managed_combine_output_compute_owner;
} DeepEpTransportCombineRequestV1;

typedef struct DeepEpTransportCombineResponseV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  DeepEpTransportTensorSlotV1 combined_x_out;
  DeepEpTransportTensorSlotV1 combined_topk_weights_out;
  DeepEpTransportEventV1 *event;
} DeepEpTransportCombineResponseV1;

typedef int32_t (*DeepEpTransportRuntimeFromPyObjectV1)(
    void *py_object, DeepEpTransportRuntimeV1 **runtime_out);
typedef int32_t (*DeepEpTransportDispatchV1)(
    const DeepEpTransportDispatchRequestV1 *request,
    DeepEpTransportDispatchResponseV1 *response);
typedef int32_t (*DeepEpTransportCombineV1)(
    const DeepEpTransportCombineRequestV1 *request,
    DeepEpTransportCombineResponseV1 *response);
typedef int32_t (*DeepEpTransportDispatchWithHandleV1)(
    const DeepEpTransportDispatchRequestV1 *request,
    DeepEpTransportDispatchResponseV1 *response);
typedef int32_t (*DeepEpTransportHandleLifetimeV1)(
    DeepEpTransportHandleV1 *handle);
typedef int32_t (*DeepEpTransportEventLifetimeV1)(
    DeepEpTransportEventV1 *event);
typedef int32_t (*DeepEpTransportEventWaitV1)(DeepEpTransportEventV1 *event);
typedef int32_t (*DeepEpTransportRuntimeLifetimeV1)(
    DeepEpTransportRuntimeV1 *runtime);
typedef int32_t (*DeepEpTransportEventFromPyObjectV1)(
    void *py_object, DeepEpTransportEventV1 **event_out);

/*
 * Keep the prefix in lock-step with
 * tools/perf_tool/probe_deepep_native_transport.py.  New optional functions
 * may only be appended and require a struct_size check by consumers.
 *
 * Ownership contract:
 * - runtime_from_pyobject returns runtime_out with one reference on success.
 * - a fresh dispatch returns handle with one reference; cached dispatch
 *   retains cached_handle and returns that additional reference.
 * - event may be NULL when DeepEP does not create an asynchronous event;
 *   otherwise it is returned with one reference.
 * - event_from_pyobject copies a DeepEP EventHandle into event_out with one
 *   reference; it never borrows the caller's PyObject.
 * - every successful retain must be paired with release.  The last release
 *   invalidates the opaque pointer.  event_wait does not consume a reference.
 * - on success, response tensor slots contain assigned intrusive references.
 *   The caller owns the slot objects.  On non-OK status, ignore all response
 *   fields and use last_error for diagnostics.
 */
typedef struct DeepEpTransportApiV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t capabilities;
  DeepEpTransportRuntimeFromPyObjectV1 runtime_from_pyobject;
  DeepEpTransportDispatchV1 dispatch;
  DeepEpTransportCombineV1 combine;
  DeepEpTransportDispatchWithHandleV1 dispatch_with_handle;
  DeepEpTransportHandleLifetimeV1 handle_retain;
  DeepEpTransportHandleLifetimeV1 handle_release;
  DeepEpTransportEventLifetimeV1 event_retain;
  DeepEpTransportEventLifetimeV1 event_release;
  DeepEpTransportEventWaitV1 event_wait;
  DeepEpTransportRuntimeLifetimeV1 runtime_retain;
  DeepEpTransportRuntimeLifetimeV1 runtime_release;
  const char *(*last_error)(void);
  uint32_t torch_cxx11_abi;
  uint32_t torch_tensor_size;
  const char *torch_version;
  DeepEpTransportEventFromPyObjectV1 event_from_pyobject;
} DeepEpTransportApiV1;

#if defined(_WIN32)
#define DEEPEP_TRANSPORT_EXPORT __declspec(dllexport)
#else
#define DEEPEP_TRANSPORT_EXPORT __attribute__((visibility("default")))
#endif

DEEPEP_TRANSPORT_EXPORT const DeepEpTransportApiV1 *
deep_ep_transport_get_api_v1(void);

#ifdef __cplusplus
}
#endif

#endif /* DEEPEP_TRANSPORT_ABI_H_ */
