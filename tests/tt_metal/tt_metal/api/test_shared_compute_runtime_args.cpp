// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/dispatch.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::shared_compute_runtime_test {

const CoreCoord core0{0, 0};
const CoreCoord core1{1, 0};

// Returns an UNPACK kernel followed by MATH and PACK kernels that borrow its runtime arguments.
ProgramDescriptor split_per_trisc() {
    ProgramDescriptor descriptor;
    for (auto processor : {ComputeProcessor::UNPACK, ComputeProcessor::MATH, ComputeProcessor::PACK}) {
        descriptor.kernels.push_back({
            .kernel_source = "void kernel_main() {}",
            .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
            .core_ranges = CoreRangeSet(CoreRange(core0, core1)),
            .config = ComputeConfigDescriptor{.processor = processor},
        });
        if (processor != ComputeProcessor::UNPACK) {
            descriptor.kernels.back().runtime_args_owner = 0;
        }
    }
    return descriptor;
}

uint32_t tensix_index() {
    return MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
}

class SharedComputeRuntimeArgs : public ::testing::Test {
protected:
    void SetUp() override { experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1); }
    void TearDown() override { experimental::disable_mock_mode(); }
};

TEST_F(SharedComputeRuntimeArgs, CPU_SplitKernelPacksRuntimeArgsLikeUnsplitKernel) {
    auto split_descriptor = split_per_trisc();
    auto& owner = split_descriptor.kernels[0];
    owner.runtime_args = {{core0, std::vector<uint32_t>(8, 7)}, {core1, std::vector<uint32_t>(4, 8)}};
    owner.common_runtime_args = {9};
    auto unsplit_owner = owner;
    std::get<ComputeConfigDescriptor>(unsplit_owner.config).processor.reset();
    // The owner shares a kernel group with this reader on core1 only, so its offsets differ per group.
    KernelDescriptor reader{
        .kernel_source = "void kernel_main() {}",
        .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
        .core_ranges = CoreRangeSet(CoreRange(core1)),
        .runtime_args = {{core1, {1, 2, 3, 4, 5}}},
        .common_runtime_args = {6},
        .config = ReaderConfigDescriptor{},
    };
    split_descriptor.kernels.push_back(reader);
    Program split(split_descriptor);
    Program unsplit(ProgramDescriptor{.kernels = {unsplit_owner, reader}});

    auto pack_runtime_args = [](Program& program) {
        auto& kernels = program.impl().get_kernels(tensix_index());
        auto& groups = program.impl().get_kernel_groups(tensix_index());
        const auto& context = MetalContext::instance();
        const auto unique =
            program_dispatch::configure_rta_offsets_for_kernel_groups(context, tensix_index(), kernels, groups, 32);
        const auto common = program_dispatch::configure_crta_offsets_for_kernel_groups(
            context, tensix_index(), kernels, groups, 32 + unique);
        return std::pair{unique, common};
    };
    EXPECT_EQ(pack_runtime_args(split), pack_runtime_args(unsplit));

    for (const auto& group : split.impl().get_kernel_groups(tensix_index())) {
        auto* baseline =
            unsplit.impl().kernels_on_core(group->core_ranges.ranges().begin()->start_coord, tensix_index());
        auto offsets = group->launch_msg.view().kernel_config().rta_offset();
        auto baseline_offsets = baseline->launch_msg.view().kernel_config().rta_offset();
        for (size_t processor = 0; processor < offsets.size(); ++processor) {
            EXPECT_EQ(offsets[processor].rta_offset(), baseline_offsets[processor].rta_offset());
            EXPECT_EQ(offsets[processor].crta_offset(), baseline_offsets[processor].crta_offset());
        }
    }
}

TEST_F(SharedComputeRuntimeArgs, CPU_BorrowerUsesOwnerRuntimeArgsAndSchema) {
    auto descriptor = split_per_trisc();
    descriptor.kernels[0].runtime_args = {{core0, {10}}, {core1, {20}}};
    descriptor.kernels[0].common_runtime_args = {30};
    descriptor.kernels[0].blaze_named_args.named_per_core_runtime_args = {{"op.scalar", {{core0, 11}, {core1, 21}}}};
    Program program(descriptor);

    EXPECT_EQ(&GetRuntimeArgs(program, 1, core0), &GetRuntimeArgs(program, 0, core0));
    EXPECT_EQ(&GetRuntimeArgs(program, 1), &GetRuntimeArgs(program, 0));
    EXPECT_EQ(&GetCommonRuntimeArgs(program, 1), &GetCommonRuntimeArgs(program, 0));
    EXPECT_EQ(
        program.impl().get_kernel(1)->named_runtime_arg_namespaces().at("op").size(),
        program.impl().get_kernel(0)->named_runtime_arg_namespaces().at("op").size());
    SetRuntimeArgs(program, 1, core0, {50, 51});
    EXPECT_EQ(GetRuntimeArgs(program, 0, core0)[0], 50);
    // Forwards to the owner, whose common runtime args are already set.
    EXPECT_THROW(SetCommonRuntimeArgs(program, 1, {1}), std::runtime_error);
}

TEST_F(SharedComputeRuntimeArgs, CPU_RejectsInvalidBorrower) {
    const auto valid = split_per_trisc();
    const std::vector<std::pair<std::string, std::function<void(ProgramDescriptor&)>>> edits = {
        {"runtime_args_owner 2 must be an earlier kernel", [](auto& d) { d.kernels[1].runtime_args_owner = 2; }},
        {"owner must be an UNPACK kernel", [](auto& d) { d.kernels[2].runtime_args_owner = 1; }},
        {"must be on the same cores as its owner",
         [](auto& d) { d.kernels[1].core_ranges = CoreRangeSet(CoreRange(core0)); }},
        {"owner must be an UNPACK kernel", [](auto& d) { d.kernels[0].config = WriterConfigDescriptor{}; }},
        {"borrower must be a MATH or PACK kernel", [](auto& d) { d.kernels[1].config = ReaderConfigDescriptor{}; }},
        {"must not supply runtime_args", [](auto& d) { d.kernels[1].runtime_args = {{core0, {1}}}; }},
        {"must not supply common_runtime_args", [](auto& d) { d.kernels[1].common_runtime_args = {1}; }},
        {"must not supply named runtime arguments",
         [](auto& d) { d.kernels[1].blaze_named_args.named_common_runtime_args = {{"op.value", 1}}; }},
        {"must not supply named runtime arguments",
         [](auto& d) { d.kernels[1].blaze_named_args.named_per_core_runtime_args = {{"op.value", {{core0, 1}}}}; }},
        {"must not supply named runtime arguments",
         [](auto& d) { d.kernels[1].blaze_named_args.named_common_runtime_arg_arrays = {{"op.array", {1, 2}}}; }},
        {"must not supply named runtime arguments",
         [](auto& d) {
             d.kernels[1].blaze_named_args.named_per_core_runtime_arg_arrays = {{"op.array", {{core0, {1, 2}}}}};
         }},
        {"must not supply buffer_bindings",
         [](auto& d) { d.kernels[1].buffer_bindings.push_back({core0, 0, nullptr}); }},
        {"must not supply common_buffer_bindings",
         [](auto& d) { d.kernels[1].common_buffer_bindings.push_back({0, nullptr}); }},
    };
    for (const auto& [expected_error, edit] : edits) {
        SCOPED_TRACE(expected_error);
        auto invalid = valid;
        edit(invalid);
        EXPECT_THAT(
            [&] { Program program(invalid); },
            ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(expected_error)));
    }
}

TEST_F(SharedComputeRuntimeArgs, CPU_MergeRebasesRuntimeArgsOwner) {
    ProgramDescriptor prefix{.kernels = {split_per_trisc().kernels[0]}};
    prefix.kernels[0].core_ranges = CoreRangeSet(CoreRange(CoreCoord{2, 0}));

    EXPECT_EQ(merge_program_descriptors({prefix, split_per_trisc()}).kernels[3].runtime_args_owner, 1);
}

using SharedComputeRuntimeArgsDevice = UnitMeshAnyDispatchFixture;

TEST_F(SharedComputeRuntimeArgsDevice, TensixAllTriscsSeeOwnerRuntimeArgsAcrossUpdates) {
    if (arch_ == tt::ARCH::QUASAR ||
        MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Emule) {
        GTEST_SKIP() << "Physical TRISC selection requires Wormhole or Blackhole silicon";
    }
    auto mesh_device = devices_.front();
    auto* device = mesh_device->get_devices()[0];
    const auto output_address = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    auto descriptor = split_per_trisc();
    auto set_owner_args = [&](uint32_t increment) {
        auto& named = descriptor.kernels[0].blaze_named_args;
        named.named_per_core_runtime_args = {{"probe.unique", {{core0, 10 + increment}, {core1, 20 + increment}}}};
        named.named_common_runtime_args = {{"probe.common", 30 + increment}};
    };
    set_owner_args(0);
    for (auto& kernel : descriptor.kernels) {
        kernel.compile_time_args = {output_address};
        kernel.kernel_source = R"(
#include "api/compute/common.h"
#include "experimental/blaze_named_args.h"

void kernel_main() {
#if defined(TRISC_UNPACK)
    constexpr uint32_t processor = 0;
#elif defined(TRISC_MATH)
    constexpr uint32_t processor = 1;
#elif defined(TRISC_PACK)
    constexpr uint32_t processor = 2;
#endif
    auto* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_compile_time_arg_val(0));
    out[processor * 2 + 0] = blaze_rt_args::get<blaze_ct_args::probe::unique>();
    out[processor * 2 + 1] = blaze_rt_args::get<blaze_ct_args::probe::common>();
}
)";
    }
    const auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    distributed::MeshWorkload workload;
    workload.add_program(device_range, Program(descriptor));
    auto& program = workload.get_programs().at(device_range);

    auto run_and_check = [&](uint32_t increment) {
        SCOPED_TRACE(increment);
        std::vector<uint32_t> cleared(6, 0);
        for (auto core : {core0, core1}) {
            ASSERT_TRUE(detail::WriteToDeviceL1(device, core, output_address, cleared));
        }
        RunProgram(mesh_device, workload);
        for (auto core : {core0, core1}) {
            std::vector<uint32_t> output;
            ASSERT_TRUE(detail::ReadFromDeviceL1(device, core, output_address, 6 * sizeof(uint32_t), output));
            EXPECT_EQ(
                output,
                std::vector<uint32_t>(
                    {10 + 10 * core.x + increment,
                     30 + increment,
                     10 + 10 * core.x + increment,
                     30 + increment,
                     10 + 10 * core.x + increment,
                     30 + increment}));
        }
    };
    ASSERT_NO_FATAL_FAILURE(run_and_check(0));

    // Update the same Program after it has run, when fast dispatch has moved its runtime args into commands.
    set_owner_args(100);
    apply_descriptor_runtime_args(program, descriptor);
    ASSERT_NO_FATAL_FAILURE(run_and_check(100));
}

}  // namespace tt::tt_metal::shared_compute_runtime_test
