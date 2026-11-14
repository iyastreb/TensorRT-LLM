/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/stridedTransfer.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include <gtest/gtest.h>

using namespace tensorrt_llm::executor::kv_cache;

namespace
{
VramRegionMap const kEmptyMap;

// Merging requires region metadata: an address that misses its region map is never coalesced.
// Tests exercising merge behavior therefore provide maps covering their addresses; a flat
// region (chunkSize=0) imposes no chunk boundaries within it.
VramRegionMap flatRegion(uintptr_t base, size_t len)
{
    VramRegionMap map;
    map[base] = {len, 0};
    return map;
}

std::pair<MemoryDescs, MemoryDescs> run(TransferDescs const& src, TransferDescs const& dst,
    VramRegionMap const& localMap = kEmptyMap, VramRegionMap const& remoteMap = kEmptyMap)
{
    return VmmDescSplitter::splitAndCoalesceTransferDescs(src, dst, localMap, remoteMap);
}
} // namespace

TEST(StridedTransferLayoutTest, ReusesGridsForReorderedAndRepeatedBlocks)
{
    detail::StridedTransferLayout layout;
    auto const regions = flatRegion(0x1000, 0x100);
    auto indices
        = layout.makeIndices(std::vector<MemoryDesc>{{0x1070, 16, 0}, {0x1010, 16, 0}, {0x1070, 16, 0}}, regions);
    ASSERT_TRUE(indices);
    EXPECT_EQ(*indices, (std::vector<int>{7, 1, 7}));
    indices = layout.findIndices(std::vector<MemoryDesc>{{0x1040, 16, 0}, {0x1020, 16, 0}, {0x1010, 16, 0}}, regions);
    ASSERT_TRUE(indices);
    EXPECT_EQ(*indices, (std::vector<int>{4, 2, 1}));
    ASSERT_EQ(layout.getRuns().size(), 1u);
    EXPECT_EQ(layout.getRuns()[0].count, 16u);
}

TEST(StridedTransferLayoutTest, SelectsAlignedBlocksWithinVmmChunks)
{
    detail::StridedTransferLayout layout;
    VramRegionMap const regions{{0x1000, {0x180, 0x100}}};
    std::vector<MemoryDesc> const descs{
        {0x10C0, 0x40, 0}, {0x1100, 0x40, 0}, {0x1010, 0x20, 0}, {0x1010, 0x20, 1}, {0x1160, 0x20, 1}};
    auto const indices = layout.makeIndices(descs, regions);
    ASSERT_TRUE(indices);
    std::vector<MemoryDesc> expanded;
    for (auto const& run : layout.getRuns())
    {
        EXPECT_LE(run.addr + (run.count - 1) * run.stride + run.len, run.addr < 0x1100 ? 0x1100u : 0x1180u);
        for (size_t index = 0; index < run.count; ++index)
        {
            expanded.emplace_back(run.addr + index * run.stride, run.len, run.deviceId);
        }
    }
    ASSERT_EQ(indices->size(), descs.size());
    for (size_t index = 0; index < descs.size(); ++index)
    {
        auto const& selected = expanded.at(indices->at(index));
        EXPECT_EQ(selected.getAddr(), descs[index].getAddr());
        EXPECT_EQ(selected.getLen(), descs[index].getLen());
        EXPECT_EQ(selected.getDeviceId(), descs[index].getDeviceId());
    }
}

TEST(StridedTransferLayoutTest, RejectsInvalidExtentsAndIndexOverflow)
{
    detail::StridedTransferLayout layout;
    VramRegionMap const regions{{0x1000, {0x200, 0x100}}};
    for (auto const& desc :
        std::vector<MemoryDesc>{{0xFFF, 1, 0}, {0x1000, 0, 0}, {0x1200, 1, 0}, {0x10F0, 32, 0}, {0x11F0, 32, 0}})
    {
        EXPECT_FALSE(layout.makeIndices(std::vector{desc}, regions));
    }
    constexpr auto kMaxAddr = std::numeric_limits<uintptr_t>::max();
    EXPECT_FALSE(layout.makeIndices(std::vector<MemoryDesc>{{kMaxAddr - 7, 4, 0}}, flatRegion(kMaxAddr - 15, 32)));
    auto const tooManyIndices = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
    EXPECT_FALSE(layout.makeIndices(std::vector<MemoryDesc>{{0x1000, 1, 0}}, flatRegion(0x1000, tooManyIndices)));
    EXPECT_TRUE(layout.getRuns().empty());
}

TEST(StridedTransferLayoutTest, IndexesEveryChunkOfTheRegionAtOnce)
{
    detail::StridedTransferLayout layout;
    VramRegionMap const regions{{0x1000, {0x180, 0x100}}};
    ASSERT_TRUE(layout.makeIndices(std::vector<MemoryDesc>{{0x1040, 0x40, 0}}, regions));
    ASSERT_EQ(layout.getRuns().size(), 2u);
    EXPECT_EQ(layout.getRuns()[1].addr, 0x1100u);
    EXPECT_EQ(layout.getRuns()[1].count, 2u);
    auto const indices = layout.findIndices(std::vector<MemoryDesc>{{0x1140, 0x40, 0}, {0x10C0, 0x40, 0}}, regions);
    ASSERT_TRUE(indices);
    EXPECT_EQ(*indices, (std::vector<int>{5, 3}));
}

TEST(StridedTransferLayoutTest, BoundsGridsPerRegionAndTheIndexSpace)
{
    constexpr auto kMaxGrids = detail::StridedTransferLayout::kMaxGridsPerRegion;
    detail::StridedTransferLayout layout;
    auto regions = flatRegion(0x1000, 0x1000);
    regions[0x9000] = {0x1000, 0};
    for (size_t phase = 0; phase < kMaxGrids; ++phase)
    {
        ASSERT_TRUE(layout.makeIndices(std::vector<MemoryDesc>{{0x1000 + phase, 64, 0}}, regions));
    }
    EXPECT_FALSE(layout.makeIndices(std::vector<MemoryDesc>{{0x1000 + kMaxGrids, 64, 0}}, regions));
    EXPECT_TRUE(layout.makeIndices(std::vector<MemoryDesc>{{0x9000, 64, 0}}, regions));
    EXPECT_EQ(layout.getRuns().size(), kMaxGrids + 1);

    detail::StridedTransferLayout large;
    auto const half = flatRegion(0x1000, static_cast<size_t>(std::numeric_limits<int>::max()) / 2 + 1);
    EXPECT_TRUE(large.makeIndices(std::vector<MemoryDesc>{{0x1000, 1, 0}}, half));
    EXPECT_FALSE(large.makeIndices(std::vector<MemoryDesc>{{0x1000, 1, 1}}, half));
}

// ==================== coalescing within flat regions (chunkSize=0) ====================

TEST(SplitAndCoalesceTest, EmptyInput)
{
    TransferDescs src{MemoryType::kVRAM, {}};
    TransferDescs dst{MemoryType::kVRAM, {}};
    auto [resSrc, resDst] = run(src, dst);
    EXPECT_EQ(resSrc.getDescs().size(), 0);
    EXPECT_EQ(resDst.getDescs().size(), 0);
}

TEST(SplitAndCoalesceTest, SinglePair)
{
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}}};
    auto [resSrc, resDst] = run(src, dst);
    ASSERT_EQ(resSrc.getDescs().size(), 1);
    ASSERT_EQ(resDst.getDescs().size(), 1);
    EXPECT_EQ(resSrc.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(resDst.getDescs()[0].getAddr(), 0x5000);
}

TEST(SplitAndCoalesceTest, BothSidesContiguous)
{
    // src contiguous AND dst contiguous within known regions — should merge into one transfer
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}, MemoryDesc{0x5100, 256, 1}}};
    auto [resSrc, resDst] = run(src, dst, flatRegion(0x1000, 0x1000), flatRegion(0x5000, 0x1000));
    ASSERT_EQ(resSrc.getDescs().size(), 1);
    ASSERT_EQ(resDst.getDescs().size(), 1);
    EXPECT_EQ(resSrc.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 512);
    EXPECT_EQ(resDst.getDescs()[0].getAddr(), 0x5000);
    EXPECT_EQ(resDst.getDescs()[0].getLen(), 512);
}

TEST(SplitAndCoalesceTest, SrcContiguousDstNot)
{
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}, MemoryDesc{0x6000, 256, 1}}};
    auto [resSrc, resDst] = run(src, dst, flatRegion(0x1000, 0x1000), flatRegion(0x5000, 0x2000));
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
}

TEST(SplitAndCoalesceTest, DstContiguousSrcNot)
{
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x2000, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}, MemoryDesc{0x5100, 256, 1}}};
    auto [resSrc, resDst] = run(src, dst, flatRegion(0x1000, 0x2000), flatRegion(0x5000, 0x1000));
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
}

TEST(SplitAndCoalesceTest, DifferentDevicesOnSrc)
{
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 1}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 0}, MemoryDesc{0x5100, 256, 0}}};
    auto [resSrc, resDst] = run(src, dst, flatRegion(0x1000, 0x1000), flatRegion(0x5000, 0x1000));
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
}

TEST(SplitAndCoalesceTest, DifferentDevicesOnDst)
{
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 0}, MemoryDesc{0x5100, 256, 1}}};
    auto [resSrc, resDst] = run(src, dst, flatRegion(0x1000, 0x1000), flatRegion(0x5000, 0x1000));
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
}

TEST(SplitAndCoalesceTest, ThreePairsAllContiguous)
{
    TransferDescs src{
        MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}, MemoryDesc{0x1200, 256, 0}}};
    TransferDescs dst{
        MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}, MemoryDesc{0x5100, 256, 1}, MemoryDesc{0x5200, 256, 1}}};
    auto [resSrc, resDst] = run(src, dst, flatRegion(0x1000, 0x1000), flatRegion(0x5000, 0x1000));
    ASSERT_EQ(resSrc.getDescs().size(), 1);
    ASSERT_EQ(resDst.getDescs().size(), 1);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 768);
    EXPECT_EQ(resDst.getDescs()[0].getLen(), 768);
}

TEST(SplitAndCoalesceTest, PartialMerge)
{
    // First two pairs merge; third pair's dst has a gap — stays separate
    TransferDescs src{
        MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}, MemoryDesc{0x1200, 256, 0}}};
    TransferDescs dst{
        MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}, MemoryDesc{0x5100, 256, 1}, MemoryDesc{0x9000, 256, 1}}};
    auto [resSrc, resDst] = run(src, dst, flatRegion(0x1000, 0x1000), flatRegion(0x5000, 0x8000));
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
    EXPECT_EQ(resSrc.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 512);
    EXPECT_EQ(resDst.getDescs()[0].getAddr(), 0x5000);
    EXPECT_EQ(resDst.getDescs()[0].getLen(), 512);
    EXPECT_EQ(resSrc.getDescs()[1].getAddr(), 0x1200);
    EXPECT_EQ(resDst.getDescs()[1].getAddr(), 0x9000);
}

TEST(SplitAndCoalesceTest, UnsortedInput)
{
    // Same as BothSidesContiguous but in reverse order — sorting by src addr should fix it
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1100, 256, 0}, MemoryDesc{0x1000, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5100, 256, 1}, MemoryDesc{0x5000, 256, 1}}};
    auto [resSrc, resDst] = run(src, dst, flatRegion(0x1000, 0x1000), flatRegion(0x5000, 0x1000));
    ASSERT_EQ(resSrc.getDescs().size(), 1);
    ASSERT_EQ(resDst.getDescs().size(), 1);
    EXPECT_EQ(resSrc.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 512);
    EXPECT_EQ(resDst.getDescs()[0].getAddr(), 0x5000);
    EXPECT_EQ(resDst.getDescs()[0].getLen(), 512);
}

TEST(SplitAndCoalesceTest, NonVramPassthrough)
{
    // Non-kVRAM descs pass through unchanged: no region info exists to bound a merge
    TransferDescs src{MemoryType::kDRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    TransferDescs dst{MemoryType::kDRAM, {MemoryDesc{0x5000, 256, 0}, MemoryDesc{0x5100, 256, 0}}};
    auto [resSrc, resDst] = run(src, dst);
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
    EXPECT_EQ(resSrc.getType(), MemoryType::kDRAM);
}

// ==================== unknown regions never merge ====================

TEST(SplitAndCoalesceTest, NoMergeWithoutRegionMetadata)
{
    // Contiguous on both sides, but neither map covers the addresses: two unknown regions are
    // indistinguishable (both look up as a miss), so merging is disabled and pairs stay separate.
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}, MemoryDesc{0x5100, 256, 1}}};
    auto [resSrc, resDst] = run(src, dst);
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
}

TEST(SplitAndCoalesceTest, NoMergeWhenRemoteRegionUnknown)
{
    // Local map covers src, but the remote side sent no region info (e.g. an older peer):
    // dst lookups miss, so nothing merges even though both sides are contiguous.
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}, MemoryDesc{0x5100, 256, 1}}};
    auto [resSrc, resDst] = run(src, dst, flatRegion(0x1000, 0x1000));
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
}

TEST(SplitAndCoalesceTest, NoMergeWhenLocalRegionUnknown)
{
    // Remote map covers dst, but src addresses miss the local map: no merging.
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}, MemoryDesc{0x5100, 256, 1}}};
    auto [resSrc, resDst] = run(src, dst, kEmptyMap, flatRegion(0x5000, 0x1000));
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
}

// ==================== chunk-boundary-constrained coalescing ====================

TEST(SplitAndCoalesceTest, MergeStopsAtSrcChunkBoundary)
{
    // Local VMM region: base=0x100000, 4MB total, 2MB chunks → boundary at 0x300000.
    VramRegionMap localMap;
    localMap[0x100000] = {0x400000, 0x200000};

    // Four contiguous 1MB pairs covering 4MB on both sides; dst is one flat remote region.
    std::vector<MemoryDesc> srcVec, dstVec;
    for (size_t i = 0; i < 4; ++i)
    {
        srcVec.emplace_back(0x100000 + i * 0x100000, 0x100000, 0);
        dstVec.emplace_back(0x900000 + i * 0x100000, 0x100000, 1);
    }
    TransferDescs src{MemoryType::kVRAM, srcVec};
    TransferDescs dst{MemoryType::kVRAM, dstVec};

    auto [resSrc, resDst] = run(src, dst, localMap, flatRegion(0x900000, 0x400000));
    // Merged per src chunk: two 2MB transfers, split exactly at the chunk boundary.
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
    EXPECT_EQ(resSrc.getDescs()[0].getAddr(), 0x100000);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 0x200000);
    EXPECT_EQ(resSrc.getDescs()[1].getAddr(), 0x300000);
    EXPECT_EQ(resSrc.getDescs()[1].getLen(), 0x200000);
    EXPECT_EQ(resDst.getDescs()[0].getAddr(), 0x900000);
    EXPECT_EQ(resDst.getDescs()[0].getLen(), 0x200000);
    EXPECT_EQ(resDst.getDescs()[1].getAddr(), 0xB00000);
    EXPECT_EQ(resDst.getDescs()[1].getLen(), 0x200000);
}

TEST(SplitAndCoalesceTest, MergeStopsAtDstChunkBoundary)
{
    // Remote VMM region: base=0x800000, 1MB chunks → boundary at 0x900000.
    VramRegionMap remoteMap;
    remoteMap[0x800000] = {0x400000, 0x100000};

    // Two contiguous pairs whose dst junction sits exactly on the remote chunk boundary.
    auto localMap = flatRegion(0x1000, 0x100000);
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 0x80000, 0}, MemoryDesc{0x81000, 0x80000, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x880000, 0x80000, 1}, MemoryDesc{0x900000, 0x80000, 1}}};

    // With a flat remote region they would merge into one transfer...
    {
        auto [resSrc, resDst] = run(src, dst, localMap, flatRegion(0x800000, 0x400000));
        ASSERT_EQ(resSrc.getDescs().size(), 1);
        ASSERT_EQ(resDst.getDescs().size(), 1);
    }
    // ...but the dst chunk boundary must block the merge.
    auto [resSrc, resDst] = run(src, dst, localMap, remoteMap);
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
    EXPECT_EQ(resDst.getDescs()[0].getAddr(), 0x880000);
    EXPECT_EQ(resDst.getDescs()[1].getAddr(), 0x900000);
}

TEST(SplitAndCoalesceTest, SplitPiecesAreNotRemerged)
{
    // A single pair spanning two src chunks stays split even though the pieces are contiguous.
    VramRegionMap localMap;
    localMap[0x100000] = {0x400000, 0x100000};

    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x100000, 0x200000, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x900000, 0x200000, 1}}};

    auto [resSrc, resDst] = run(src, dst, localMap, flatRegion(0x900000, 0x400000));
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 0x100000);
    EXPECT_EQ(resSrc.getDescs()[1].getLen(), 0x100000);
}

TEST(SplitAndCoalesceTest, UnalignedRegionBaseBoundary)
{
    // Chunk boundaries are relative to the region base, not absolute alignment.
    // base=0x180000, 1MB chunks → boundaries at 0x280000, 0x380000, ...
    VramRegionMap localMap;
    localMap[0x180000] = {0x300000, 0x100000};

    // Three contiguous 512KB pairs: first two share the first chunk, third starts a new chunk.
    std::vector<MemoryDesc> srcVec, dstVec;
    for (size_t i = 0; i < 3; ++i)
    {
        srcVec.emplace_back(0x180000 + i * 0x80000, 0x80000, 0);
        dstVec.emplace_back(0x900000 + i * 0x80000, 0x80000, 1);
    }
    TransferDescs src{MemoryType::kVRAM, srcVec};
    TransferDescs dst{MemoryType::kVRAM, dstVec};

    auto [resSrc, resDst] = run(src, dst, localMap, flatRegion(0x900000, 0x300000));
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    EXPECT_EQ(resSrc.getDescs()[0].getAddr(), 0x180000);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 0x100000);
    EXPECT_EQ(resSrc.getDescs()[1].getAddr(), 0x280000);
    EXPECT_EQ(resSrc.getDescs()[1].getLen(), 0x80000);
}

TEST(SplitAndCoalesceTest, NoMergeAcrossRegions)
{
    // Two VA-adjacent but distinct local regions (cudaMalloc-style, chunkSize=0):
    // contiguous descs must not merge across the region boundary.
    VramRegionMap localMap;
    localMap[0x100000] = {0x100000, 0};
    localMap[0x200000] = {0x100000, 0};

    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x100000, 0x100000, 0}, MemoryDesc{0x200000, 0x100000, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x900000, 0x100000, 1}, MemoryDesc{0xA00000, 0x100000, 1}}};

    auto [resSrc, resDst] = run(src, dst, localMap, flatRegion(0x900000, 0x200000));
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
}

TEST(SplitAndCoalesceTest, MergeWithinSingleRegion)
{
    // Control for NoMergeAcrossRegions: same layout as one region (chunkSize=0) merges freely.
    VramRegionMap localMap;
    localMap[0x100000] = {0x200000, 0};

    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x100000, 0x100000, 0}, MemoryDesc{0x200000, 0x100000, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x900000, 0x100000, 1}, MemoryDesc{0xA00000, 0x100000, 1}}};

    auto [resSrc, resDst] = run(src, dst, localMap, flatRegion(0x900000, 0x200000));
    ASSERT_EQ(resSrc.getDescs().size(), 1);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 0x200000);
    EXPECT_EQ(resDst.getDescs()[0].getLen(), 0x200000);
}

TEST(SplitAndCoalesceTest, NoMergeAcrossRemoteRegions)
{
    // The remote side registered two discrete but VA-adjacent buffers (chunkSize=0 each):
    // contiguous dst descs must not merge across the remote registration boundary.
    VramRegionMap remoteMap;
    remoteMap[0x900000] = {0x100000, 0};
    remoteMap[0xA00000] = {0x100000, 0};

    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x100000, 0x100000, 0}, MemoryDesc{0x200000, 0x100000, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x900000, 0x100000, 1}, MemoryDesc{0xA00000, 0x100000, 1}}};

    auto [resSrc, resDst] = run(src, dst, flatRegion(0x100000, 0x200000), remoteMap);
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
}

TEST(SplitAndCoalesceTest, CoalesceDisabledSplitsOnly)
{
    // With coalescing disabled, contiguous pairs stay separate and chunk splitting still applies.
    VramRegionMap localMap;
    localMap[0x100000] = {0x400000, 0x100000};

    // Two contiguous 512KB pairs within one chunk plus one pair spanning a chunk boundary.
    TransferDescs src{MemoryType::kVRAM,
        {MemoryDesc{0x100000, 0x80000, 0}, MemoryDesc{0x180000, 0x80000, 0}, MemoryDesc{0x200000, 0x200000, 0}}};
    TransferDescs dst{MemoryType::kVRAM,
        {MemoryDesc{0x900000, 0x80000, 1}, MemoryDesc{0x980000, 0x80000, 1}, MemoryDesc{0xA00000, 0x200000, 1}}};

    auto [resSrc, resDst]
        = VmmDescSplitter::splitAndCoalesceTransferDescs(src, dst, localMap, kEmptyMap, /*enableCoalesce=*/false);
    // No merging: pair 1, pair 2, and pair 3 split into two chunk pieces → 4 descs.
    ASSERT_EQ(resSrc.getDescs().size(), 4);
    ASSERT_EQ(resDst.getDescs().size(), 4);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 0x80000);
    EXPECT_EQ(resSrc.getDescs()[1].getLen(), 0x80000);
    EXPECT_EQ(resSrc.getDescs()[2].getLen(), 0x100000);
    EXPECT_EQ(resSrc.getDescs()[3].getLen(), 0x100000);
}

TEST(SplitAndCoalesceTest, CoalesceDisabledPreservesInputOrder)
{
    // With coalescing disabled there is no sorting either: descs come out in input order,
    // matching the historical split-only behavior.
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x2000, 256, 0}, MemoryDesc{0x1000, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x6000, 256, 1}, MemoryDesc{0x5000, 256, 1}}};

    auto [resSrc, resDst]
        = VmmDescSplitter::splitAndCoalesceTransferDescs(src, dst, kEmptyMap, kEmptyMap, /*enableCoalesce=*/false);
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    EXPECT_EQ(resSrc.getDescs()[0].getAddr(), 0x2000);
    EXPECT_EQ(resSrc.getDescs()[1].getAddr(), 0x1000);
    EXPECT_EQ(resDst.getDescs()[0].getAddr(), 0x6000);
    EXPECT_EQ(resDst.getDescs()[1].getAddr(), 0x5000);
}

TEST(SplitAndCoalesceTest, SplitAndMergeScatteredBlocks)
{
    // Scattered input order + chunked src region: blocks are sorted, merged per chunk.
    // base=0x100000, 1MB chunks; four 512KB blocks given out of order.
    VramRegionMap localMap;
    localMap[0x100000] = {0x400000, 0x100000};

    std::vector<size_t> perm{2, 0, 3, 1};
    std::vector<MemoryDesc> srcVec, dstVec;
    for (size_t i : perm)
    {
        srcVec.emplace_back(0x100000 + i * 0x80000, 0x80000, 0);
        dstVec.emplace_back(0x900000 + i * 0x80000, 0x80000, 1);
    }
    TransferDescs src{MemoryType::kVRAM, srcVec};
    TransferDescs dst{MemoryType::kVRAM, dstVec};

    auto [resSrc, resDst] = run(src, dst, localMap, flatRegion(0x900000, 0x400000));
    // 2MB of contiguous data over two 1MB chunks → one transfer per chunk.
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    EXPECT_EQ(resSrc.getDescs()[0].getAddr(), 0x100000);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 0x100000);
    EXPECT_EQ(resSrc.getDescs()[1].getAddr(), 0x200000);
    EXPECT_EQ(resSrc.getDescs()[1].getLen(), 0x100000);
    EXPECT_EQ(resDst.getDescs()[0].getAddr(), 0x900000);
    EXPECT_EQ(resDst.getDescs()[1].getAddr(), 0xA00000);
}
