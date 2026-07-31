#include <deep_ep/impls/dispatch_copy_epilogue.cuh>

using namespace deep_ep::elastic;

static void instantiate_direct_backward_epilogue() {
    auto ptr = reinterpret_cast<void*>(
        &dispatch_copy_epilogue_impl<
            false, true, false, true,
            148, 148, 8,
            1, 2,
            8192, 0,
            4096,
            128, 10, 256>);
    (void)ptr;
}
