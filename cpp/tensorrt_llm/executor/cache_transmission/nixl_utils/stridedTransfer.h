// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorrt_llm::executor::kv_cache::detail
{

struct StridedTransferRun
{
    uintptr_t addr;
    size_t len;
    uint32_t deviceId;
    size_t stride;
    size_t count;
};

class StridedTransferLayout
{
public:
    //! Grids appended per region; descriptors that would need more are not selected.
    static constexpr size_t kMaxGridsPerRegion = 16;

    [[nodiscard]] std::vector<StridedTransferRun> const& getRuns() const
    {
        return mRuns;
    }

    template <typename Desc, typename RegionMap>
    [[nodiscard]] std::optional<std::vector<int>> findIndices(
        std::vector<Desc> const& descs, RegionMap const& regions) const
    {
        std::vector<int> indices;
        indices.reserve(descs.size());
        for (auto const& desc : descs)
        {
            auto const run = getRun(desc, regions);
            if (!run)
            {
                return std::nullopt;
            }
            auto const found = mOffsets.find(std::make_tuple(run->addr, run->len, run->deviceId, run->count));
            if (found == mOffsets.end())
            {
                return std::nullopt;
            }
            indices.push_back(static_cast<int>(found->second + (desc.getAddr() - run->addr) / run->stride));
        }
        return indices;
    }

    //! Select descriptors from registered, chunk-bounded grids; append missing grids without reindexing.
    template <typename Desc, typename RegionMap>
    [[nodiscard]] std::optional<std::vector<int>> makeIndices(std::vector<Desc> const& descs, RegionMap const& regions)
    {
        std::vector<int> indices;
        indices.reserve(descs.size());
        for (auto const& desc : descs)
        {
            auto const run = getRun(desc, regions);
            if (!run)
            {
                return std::nullopt;
            }
            auto const key = std::make_tuple(run->addr, run->len, run->deviceId, run->count);
            auto found = mOffsets.find(key);
            if (found == mOffsets.end())
            {
                if (!addGrid(*run, regions))
                {
                    return std::nullopt;
                }
                found = mOffsets.find(key);
            }
            indices.push_back(static_cast<int>(found->second + (desc.getAddr() - run->addr) / run->stride));
        }
        return indices;
    }

private:
    template <typename Desc, typename RegionMap>
    [[nodiscard]] static std::optional<StridedTransferRun> getRun(Desc const& desc, RegionMap const& regions)
    {
        auto const addr = desc.getAddr();
        auto const len = desc.getLen();
        auto region = regions.upper_bound(addr);
        if (len == 0 || region == regions.begin())
        {
            return std::nullopt;
        }
        --region;
        auto const base = region->first;
        auto const& info = region->second;
        if (info.totalLen > std::numeric_limits<uintptr_t>::max() - base || addr - base >= info.totalLen)
        {
            return std::nullopt;
        }
        auto const chunkSize = info.chunkSize == 0 ? info.totalLen : info.chunkSize;
        auto const chunkOffset = (addr - base) / chunkSize * chunkSize;
        auto const chunkBase = base + chunkOffset;
        auto const chunkLen = std::min(chunkSize, info.totalLen - chunkOffset);
        if (len > chunkLen - (addr - chunkBase))
        {
            return std::nullopt;
        }
        auto const start = chunkBase + (addr - chunkBase) % len;
        auto const count = (chunkLen - (start - chunkBase)) / len;
        return StridedTransferRun{start, len, desc.getDeviceId(), len, count};
    }

    //! Append the grid of `run` in every chunk of its region, so that a block geometry is prepared once.
    template <typename RegionMap>
    [[nodiscard]] bool addGrid(StridedTransferRun const& run, RegionMap const& regions)
    {
        auto region = regions.upper_bound(run.addr);
        --region;
        auto const base = region->first;
        auto const totalLen = region->second.totalLen;
        auto const chunkSize = region->second.chunkSize == 0 ? totalLen : region->second.chunkSize;
        auto const phase = (run.addr - base) % chunkSize;
        auto& grids = mGridsPerRegion[base];
        if (grids == kMaxGridsPerRegion
            || totalLen / run.len > static_cast<size_t>(std::numeric_limits<int>::max()) - mSize)
        {
            return false;
        }
        ++grids;
        for (size_t offset = 0, chunkLen = 0; offset < totalLen; offset += chunkLen)
        {
            chunkLen = std::min(chunkSize, totalLen - offset);
            if (chunkLen >= phase + run.len)
            {
                auto const count = (chunkLen - phase) / run.len;
                mOffsets.emplace(std::make_tuple(base + offset + phase, run.len, run.deviceId, count), mSize);
                mRuns.push_back({base + offset + phase, run.len, run.deviceId, run.len, count});
                mSize += count;
            }
        }
        return true;
    }

    std::vector<StridedTransferRun> mRuns;
    std::map<std::tuple<uintptr_t, size_t, uint32_t, size_t>, size_t> mOffsets;
    std::map<uintptr_t, size_t> mGridsPerRegion;
    size_t mSize = 0;
};

} // namespace tensorrt_llm::executor::kv_cache::detail
