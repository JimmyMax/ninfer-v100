#pragma once

// KVMem sparse working-set bridge: a faithful port of the official ninfer-kvmem integration
// (kvmem-llama.cpp v0.18.0-ninfer-rc2, backends/patches/ninfer-kvmem.patch) onto this tree's
// ProgramImplCore. Included from instantiate.h after program_impl.h; defines ProgramImplCore
// members declared in program.h.
//
// Position domains: sequence-level APIs (ledger, text_kv_valid, prepare_window, checkpoints)
// use LOGICAL positions. The device address space uses COMPACT positions (logical minus
// window.removed_tokens) through compact_position(). Attention cache slots are compact while
// RoPE positions stay logical (rope_delta absorbs the shift at the execution layer).

#include <kvmem/mean_key_retrieval.hpp>

#include <atomic>
#include <cstdio>
#include <cstring>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr std::uint32_t kvmem_page_tokens = static_cast<std::uint32_t>(kPagedKVPageSize);
std::atomic<std::uint64_t> next_window_session{1};

std::uint32_t kvmem_valid_columns(std::uint32_t logical_page, std::uint32_t frontier) {
    return std::min(kvmem_page_tokens, frontier - logical_page * kvmem_page_tokens);
}

} // namespace

// This bridge borrows Program's existing native pool, address space, stream and reservation.
// It owns no device allocator. Its lifetime is one safe boundary; durable history and versions
// travel with SequenceState.
class WindowMemoryBackend final : public kvmem::MemoryBackend {
public:
    WindowMemoryBackend(ProgramImplCore& program, SequenceState& sequence, std::uint32_t end)
        : program(program), sequence(sequence), end(end),
          layout(plan_host_kv_page_layout(program.text_kv_pages->physical_pool().geometry())) {
        if (program.backend_kv_pages) {
            if (sequence.mtp_kv_valid != sequence.text_kv_valid) {
                throw std::logic_error(
                    "KVMem dual-pool transfer requires a committed safe boundary");
            }
            mtp_layout =
                plan_host_kv_page_layout(program.backend_kv_pages->physical_pool().geometry());
        }
        page_bytes = layout.page_stride + (mtp_layout ? mtp_layout->page_stride : 0);
        auto& stamp = sequence.window.stamp;
        if (!stamp.session.id) {
            stamp = {true, {next_window_session.fetch_add(1), 1}, 1, 0, 0, 1};
        }
        if (!stamp.valid) { throw std::logic_error("KVMem sequence requires fresh prefill"); }
        if (stamp.frontier != sequence.text_kv_valid) {
            stamp.frontier = sequence.text_kv_valid;
            ++stamp.history_revision;
        }
        const auto count = kv_pages_for_tokens(sequence.text_kv_valid);
        sequence.window.archive.resize(count);
        sequence.window.host_versions.resize(count);
    }

    kvmem::MemoryDescriptor descriptor() const override {
        kvmem::MemoryDescriptor d;
        d.identity = {"ninfer",
            program.backend_kv_pages ? "kvmem-main-mtp-pages-v2" : "kvmem-native-pages-v1",
            "qwen3.6-27b", "qwen3.6-kvmem-state-image-v1"};
        const char* layout_name = nullptr;
        switch (program.kv_storage) {
        case KvCacheStorage::BFloat16: layout_name = "ninfer.bf16.page.v1"; break;
        case KvCacheStorage::Int8Group64: layout_name = "ninfer.int8-g64.page.v1"; break;
        case KvCacheStorage::Fp8E4M3Row256: layout_name = "ninfer.fp8-e4m3-row256.page.v1"; break;
        case KvCacheStorage::Nvfp4Group16: layout_name = "ninfer.nvfp4-g16.page.v1"; break;
        case KvCacheStorage::Fp8KeyNvfp4Value:
            layout_name = "ninfer.k8v4-fp8-nvfp4.page.v1";
            break;
        default: throw std::logic_error("Unsupported KVMem native KV storage");
        }
        d.layout = {layout_name, 1, kvmem_page_tokens, page_bytes, 256, {}};
        const std::uint32_t planes = program.kv_storage == KvCacheStorage::BFloat16 ? 2 : 4;
        const kvmem::PayloadPlaneKind kinds[] = {kvmem::PayloadPlaneKind::Key,
            kvmem::PayloadPlaneKind::Value, kvmem::PayloadPlaneKind::KeyScale,
            kvmem::PayloadPlaneKind::ValueScale};
        std::size_t ordinal = 0;
        for (std::uint32_t layer = 0; layer < static_cast<std::uint32_t>(TextConfig::layers);
             ++layer) {
            if (!TextConfig::is_full_attention(static_cast<int>(layer))) { continue; }
            for (std::uint32_t kind = 0; kind < planes; ++kind) {
                const auto& plane = layout.planes.at(ordinal++);
                d.layout.planes.push_back(
                    {layer, kinds[kind], plane.offset, plane.page_payload_bytes, 1, 0});
            }
        }
        if (ordinal != layout.planes.size()) {
            throw std::logic_error("KVMem plane inventory mismatch");
        }
        if (mtp_layout) {
            for (std::uint32_t kind = 0; kind < planes; ++kind) {
                const auto& plane = mtp_layout->planes.at(kind);
                d.layout.planes.push_back({static_cast<std::uint32_t>(TextConfig::layers),
                    kinds[kind], layout.page_stride + plane.offset, plane.page_payload_bytes, 1,
                    0});
            }
        }
        d.capabilities = {kvmem_page_tokens, program.capacity, true, true, true, false, false,
                          false};
        return d;
    }

    kvmem::MemorySnapshot snapshot() const override {
        kvmem::MemorySnapshot s;
        s.stamp = stamp();
        const auto count = kv_pages_for_tokens(sequence.text_kv_valid);
        for (std::uint32_t id = 0; id < count; ++id) {
            const auto columns = kvmem_valid_columns(id, sequence.text_kv_valid);
            s.blocks.push_back({{s.stamp.session, id, id * kvmem_page_tokens, columns, columns, 0},
                sequence.window.host_versions.at(id)});
        }
        for (auto id : sequence.window.pages) {
            if (id < count) { s.resident.push_back(id); }
        }
        return s;
    }
    kvmem::MemoryStamp stamp() const noexcept override { return sequence.window.stamp; }
    std::unique_ptr<kvmem::MemoryTransfer> prepare(const kvmem::WorkingSetPlan&) override;
    void invalidate() noexcept override { sequence.window.stamp.valid = false; }

    ProgramImplCore& program;
    SequenceState& sequence;
    std::uint32_t end;
    HostKVPageLayout layout;
    std::optional<HostKVPageLayout> mtp_layout;
    std::size_t page_bytes = 0;
};

class WindowMemoryTransfer final : public kvmem::MemoryTransfer {
public:
    WindowMemoryTransfer(WindowMemoryBackend& backend, const kvmem::WorkingSetPlan& plan,
                         bool verify_restore = true)
        : backend(backend), plan(plan) {
        auto& seq = backend.sequence;
        auto& p = backend.program;
        const auto count = kv_pages_for_tokens(seq.text_kv_valid);
        pending.resize(count);
        // Rebuilding a native compact address space also moves retained pages.
        // Reserve their host records up front and include them in the hard H bound.
        // Complete host copies are reused; only the changed tail is copied again.
        std::uint64_t bytes = seq.window.host_bytes;
        for (std::uint32_t slot = 0; slot < seq.window.pages.size(); ++slot) {
            const auto id = seq.window.pages[slot];
            if (id >= count) { continue; }
            if (!seq.window.archive[id]) {
                if (bytes > p.kvmem_options.host_bytes ||
                    backend.page_bytes > p.kvmem_options.host_bytes - bytes) {
                    throw kvmem::MemoryError(kvmem::MemoryErrorCode::BudgetExceeded,
                        "KVMem host payload budget exhausted before transfer");
                }
                bytes += backend.page_bytes;
                pending[id] = std::make_unique<KvmemHostRecord>(backend.page_bytes);
            }
            if (seq.window.host_versions[id] != kvmem_valid_columns(id, seq.text_kv_valid)) {
                spill_ids.push_back(id);
                spill_slots.push_back(slot);
                sources.push_back(p.text_kv_addresses->physical_page(seq.kv->text, slot));
                source_epochs.push_back(p.text_kv_addresses->content_epoch(seq.kv->text, slot));
                if (backend.mtp_layout) {
                    mtp_sources.push_back(
                        p.backend_kv_addresses->physical_page(*seq.kv->backend, slot));
                    mtp_source_epochs.push_back(
                        p.backend_kv_addresses->content_epoch(*seq.kv->backend, slot));
                }
            }
        }
        prepared_host_bytes = bytes;
        for (const auto& block : plan.next.blocks) {
            const auto id = static_cast<std::uint32_t>(block.block.id);
            if (!record(id)) { throw std::logic_error("KVMem selected page has no native payload"); }
            selected.push_back(id);
        }
        if (seq.text_kv_valid % kvmem_page_tokens &&
            (selected.empty() || selected.back() != seq.text_kv_valid / kvmem_page_tokens)) {
            throw std::logic_error("KVMem partial append page must remain the last selected page");
        }
        const auto rows = std::max(spill_ids.size(), selected.size());
        if (rows > p.kvmem_window_tokens / kvmem_page_tokens) {
            throw std::logic_error("KVMem staging exceeds its physical window");
        }
        const auto staging_bytes = rows * backend.page_bytes;
        if (staging_bytes && (!p.memory_kv_staging || p.memory_kv_staging->size() < staging_bytes)) {
            p.memory_kv_staging = std::make_unique<PinnedHostBuffer>(staging_bytes);
        }
        // Only this bounded buffer participates in CUDA host transfers. Archive
        // records are byte-preserving pageable RAM and are never CUDA-registered.
        auto* staging = p.memory_kv_staging
            ? static_cast<std::byte*>(p.memory_kv_staging->data()) : nullptr;
        if (staging_bytes) std::memset(staging, 0, staging_bytes);
        for (std::size_t i = 0; i < spill_ids.size(); ++i)
            spill_records.push_back(staging + i * backend.page_bytes);
        for (std::size_t i = 0; i < selected.size(); ++i)
            restore_records.push_back(staging + i * backend.page_bytes);
        if (verify_restore && p.kvmem_options.verify_transfers) {
            const auto bytes_state = p.state_images->host_layout().image_bytes;
            before = std::make_unique<PinnedHostBuffer>(bytes_state);
            after = std::make_unique<PinnedHostBuffer>(bytes_state);
            std::memset(before->data(), 0, bytes_state);
            std::memset(after->data(), 0, bytes_state);
            checked = std::make_unique<PinnedHostBuffer>(selected.size() * backend.page_bytes);
            std::memset(before->data(), 0, bytes_state);
            std::memset(after->data(), 0, bytes_state);
            std::memset(checked->data(), 0, selected.size() * backend.page_bytes);
        }
    }

    void start_spill() override {
        auto& p = backend.program;
        auto& seq = backend.sequence;
        p.device.synchronize();
        if (before) {
            p.state_images->copy_to_host(p.state_store->physical_slot(seq.state.write),
                qwen3_6::HostStateImageView{static_cast<std::byte*>(before->data()),
                                            &p.state_images->host_layout()},
                p.device.transfer_stream);
        }
        p.text_kv_pages->physical_pool().copy_to_host_records(sources, spill_records, spill_ids,
            backend.layout, p.device.transfer_stream);
        if (backend.mtp_layout) {
            std::vector<std::byte*> mtp_records;
            for (auto* record : spill_records)
                mtp_records.push_back(record + backend.layout.page_stride);
            p.backend_kv_pages->physical_pool().copy_to_host_records(mtp_sources, mtp_records,
                spill_ids, *backend.mtp_layout, p.device.transfer_stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(p.device.transfer_stream));
        for (std::size_t i = 0; i < spill_ids.size(); ++i) {
            if (source_epochs[i] !=
                p.text_kv_addresses->content_epoch(seq.kv->text, spill_slots[i])) {
                throw std::logic_error("KVMem source epoch changed during spill");
            }
            if (backend.mtp_layout && mtp_source_epochs[i] !=
                p.backend_kv_addresses->content_epoch(*seq.kv->backend, spill_slots[i])) {
                throw std::logic_error("KVMem MTP source epoch changed during spill");
            }
            std::memcpy(record(spill_ids[i])->data(), spill_records[i], backend.page_bytes);
        }
        for (std::size_t id = 0; id < pending.size(); ++id) {
            if (pending[id]) { seq.window.archive[id] = std::move(pending[id]); }
        }
        for (auto id : spill_ids) {
            seq.window.host_versions[id] = kvmem_valid_columns(id, seq.text_kv_valid);
        }
        seq.window.host_bytes = prepared_host_bytes;
        seq.window.spilled_bytes += spill_ids.size() * backend.page_bytes;
    }

    kvmem::TransferPoll poll() override { return {kvmem::TransferStatus::Complete, {}}; }

    void start_restore() override {
        auto& p = backend.program;
        auto& seq = backend.sequence;
        for (std::size_t i = 0; i < selected.size(); ++i) {
            std::memcpy(static_cast<std::byte*>(p.memory_kv_staging->data()) +
                            i * backend.page_bytes,
                        record(selected[i])->data(), backend.page_bytes);
        }
        destructive = true;
        p.text_kv_addresses->destructive_truncate(seq.kv->text, 0);
        const auto compact_end = static_cast<std::uint32_t>(plan.next.visible_tokens) +
                                 backend.end - seq.text_kv_valid;
        p.text_kv_addresses->ensure_mapped_to_tokens(seq.kv->text, compact_end,
                                                     p.device.transfer_stream);
        if (backend.mtp_layout) {
            p.backend_kv_addresses->destructive_truncate(*seq.kv->backend, 0);
            p.backend_kv_addresses->ensure_mapped_to_tokens(*seq.kv->backend, compact_end,
                                                            p.device.transfer_stream);
        }
        std::vector<DeviceKVPageHandle> destinations;
        for (std::uint32_t i = 0; i < selected.size(); ++i) {
            destinations.push_back(p.text_kv_addresses->physical_page(seq.kv->text, i));
        }
        auto& pool = p.text_kv_pages->physical_pool();
        pool.copy_from_host_records(restore_records, selected, destinations, backend.layout,
                                    p.device.transfer_stream);
        std::vector<DeviceKVPageHandle> mtp_destinations;
        if (backend.mtp_layout) {
            std::vector<const std::byte*> mtp_records;
            for (std::size_t i = 0; i < selected.size(); ++i) {
                mtp_destinations.push_back(p.backend_kv_addresses->physical_page(
                    *seq.kv->backend, static_cast<std::uint32_t>(i)));
                mtp_records.push_back(restore_records[i] + backend.layout.page_stride);
            }
            p.backend_kv_pages->physical_pool().copy_from_host_records(mtp_records, selected,
                mtp_destinations, *backend.mtp_layout, p.device.transfer_stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(p.device.transfer_stream));
        if (checked) {
            std::vector<std::byte*> rows;
            for (std::size_t i = 0; i < selected.size(); ++i) {
                rows.push_back(static_cast<std::byte*>(checked->data()) + i * backend.page_bytes);
            }
            pool.copy_to_host_records(destinations, rows, {}, backend.layout,
                                      p.device.transfer_stream);
            if (backend.mtp_layout) {
                std::vector<std::byte*> mtp_rows;
                for (auto* record : rows) mtp_rows.push_back(record + backend.layout.page_stride);
                p.backend_kv_pages->physical_pool().copy_to_host_records(mtp_destinations, mtp_rows,
                    {}, *backend.mtp_layout, p.device.transfer_stream);
            }
            p.state_images->copy_to_host(p.state_store->physical_slot(seq.state.write),
                qwen3_6::HostStateImageView{static_cast<std::byte*>(after->data()),
                                            &p.state_images->host_layout()},
                p.device.transfer_stream);
            CUDA_CHECK(cudaStreamSynchronize(p.device.transfer_stream));
            for (std::size_t i = 0; i < rows.size(); ++i) {
                if (std::memcmp(rows[i], restore_records[i], backend.page_bytes)) {
                    throw std::logic_error("KVMem native payload changed during restore");
                }
            }
            if (std::memcmp(before->data(), after->data(),
                            p.state_images->host_layout().image_bytes)) {
                throw std::logic_error("KVMem page transfer changed recurrent state");
            }
        }
        seq.window.restored_bytes += selected.size() * backend.page_bytes;
    }

    void publish(const kvmem::ExecutionView& view) override {
        auto& p = backend.program;
        auto& seq = backend.sequence;
        p.text_kv_addresses->commit_frontier(seq.kv->text,
            static_cast<std::uint32_t>(view.visible_tokens));
        if (backend.mtp_layout) {
            p.backend_kv_addresses->commit_frontier(*seq.kv->backend,
                static_cast<std::uint32_t>(view.visible_tokens));
            // The earlier proposal was generated against the previous view. Rebuild
            // it from the next verified anchor instead of publishing stale drafts.
            seq.mtp_draft_count = 0;
        }
        seq.window.removed_tokens = seq.text_kv_valid - static_cast<std::uint32_t>(view.visible_tokens);
        seq.window.pages = std::move(selected);
        seq.window.stamp = view.stamp;
        ++seq.window.swaps;
        published = true;
        if (p.kvmem_options.verify_transfers) {
            std::fprintf(stderr,
                "KVMEM_NATIVE_VIEW logical=%u compact=%llu physical_pages=%u host_bytes=%llu "
                "swap=%u payload_exact=1 state_exact=1\n",
                seq.text_kv_valid, static_cast<unsigned long long>(view.visible_tokens),
                p.text_kv_pages->physical_pool().allocated_pages(),
                static_cast<unsigned long long>(seq.window.host_bytes), seq.window.swaps);
        }
    }

    kvmem::AbortResult abort() noexcept override {
        try {
            backend.program.device.synchronize();
            CUDA_CHECK(cudaStreamSynchronize(backend.program.device.transfer_stream));
        }
        catch (...) { backend.invalidate(); return kvmem::AbortResult::SessionInvalidated; }
        if (destructive && !published) {
            backend.invalidate();
            return kvmem::AbortResult::SessionInvalidated;
        }
        return kvmem::AbortResult::PreviousViewPreserved;
    }

private:
    KvmemHostRecord* record(std::uint32_t id) const {
        return pending.at(id) ? pending[id].get() : backend.sequence.window.archive.at(id).get();
    }
    WindowMemoryBackend& backend;
    kvmem::WorkingSetPlan plan;
    std::vector<std::unique_ptr<KvmemHostRecord>> pending;
    std::vector<std::uint32_t> spill_ids, spill_slots, selected;
    std::vector<DeviceKVPageHandle> sources, mtp_sources;
    std::vector<std::uint64_t> mtp_source_epochs;
    std::vector<std::uint64_t> source_epochs;
    std::vector<std::byte*> spill_records;
    std::vector<const std::byte*> restore_records;
    std::unique_ptr<PinnedHostBuffer> before, after, checked;
    std::uint64_t prepared_host_bytes = 0;
    bool destructive = false, published = false;
};

std::unique_ptr<kvmem::MemoryTransfer> WindowMemoryBackend::prepare(
    const kvmem::WorkingSetPlan& plan) {
    if (!(plan.expected == stamp())) { throw std::logic_error("KVMem plan is stale"); }
    return std::make_unique<WindowMemoryTransfer>(*this, plan);
}

void ProgramImplCore::prepare_window(SequenceState& sequence, std::uint32_t begin,
                                     std::uint32_t end, bool force_selection) {
    if (!kvmem_window_tokens) { return; }
    if (!sequence.kv || begin != sequence.text_kv_valid || end < begin || end > capacity ||
        end - begin > kvmem_options.reserve_tokens) {
        throw std::logic_error("KVMem append exceeds logical frontier or reserve");
    }
    if (end == begin && !force_selection) { return; }
    if (force_selection || compact_position(sequence, end) > kvmem_window_tokens) {
        auto backend = std::make_shared<WindowMemoryBackend>(*this, sequence, end);
        kvmem::MemorySession session(backend);
        const auto count = kv_pages_for_tokens(begin);
        const auto keep = std::min(count, kvmem_options.selected_tokens / kvmem_page_tokens);
        auto& state = sequence.window;
        std::vector<std::uint64_t> mandatory;
        if (count == 0) {
            mandatory.push_back(0);
        } else {
            const auto sink_keep = std::min(kvmem_options.sink_blocks(), count);
            for (std::uint32_t id = 0; id < sink_keep; ++id) mandatory.push_back(id);
            for (const auto id : state.media_pages) if (id < count) mandatory.push_back(id);
            const auto recent_keep = std::min(kvmem_options.recent_blocks(), count);
            for (std::uint32_t i = 0; i < recent_keep; ++i) mandatory.push_back(count - 1 - i);
        }
        std::sort(mandatory.begin(), mandatory.end());
        mandatory.erase(std::unique(mandatory.begin(), mandatory.end()), mandatory.end());
        std::vector<std::uint64_t> selected;
        const bool query_complete = state.query_tokens != 0 &&
            (state.query_replayed || begin >= state.query_end) &&
            state.query_tokens == state.query_end - state.query_begin;
        if (query_complete) {
            if (!state.selector || state.statistics_frontier != begin ||
                state.selector->total_tokens() != begin) {
                throw std::logic_error(
                    "KVMem retrieval index does not match the evaluated prefix");
            }
            std::vector<std::uint32_t> counts(
                static_cast<std::size_t>(TextConfig::full_attention_layers()), state.query_tokens);
            state.selector->set_retrieval_scores(kvmem::mean_key_retrieval_scores(
                *state.statistics, count,
                {state.query_sum.data(), counts.data(),
                 static_cast<std::uint32_t>(TextConfig::full_attention_layers()),
                 static_cast<std::uint32_t>(TextConfig::head_dim),
                 static_cast<std::uint32_t>(TextConfig::query_heads),
                 static_cast<std::uint32_t>(TextConfig::kv_heads)}));
            const std::vector<std::uint32_t> required(mandatory.begin(), mandatory.end());
            const auto ids = state.selector->pick_topk_blocks(required);
            selected.assign(ids.begin(), ids.end());
            if (kvmem_options.verify_transfers) {
                std::size_t cold = 0;
                for (auto id : ids) {
                    if (std::find(state.pages.begin(), state.pages.end(), id) ==
                        state.pages.end()) {
                        ++cold;
                    }
                }
                std::fprintf(stderr, "KVMEM_RETRIEVAL logical=%u query_tokens=%u cold=%zu selected=",
                             begin, state.query_tokens, cold);
                for (std::size_t i = 0; i < ids.size(); ++i)
                    std::fprintf(stderr, "%s%u", i ? "," : "", ids[i]);
                std::fprintf(stderr, "\n");
            }
        } else {
            selected = mandatory;
            for (auto id = count; id-- > 0 && selected.size() < keep;) {
                if (std::find(selected.begin(), selected.end(), id) == selected.end()) {
                    selected.push_back(id);
                }
            }
            std::sort(selected.begin(), selected.end());
        }
        session.prepare(backend->stamp(), selected, mandatory,
            {kvmem_options.selected_tokens, kvmem_options.reserve_tokens, kvmem_window_tokens,
             kvmem_options.host_bytes});
        while (session.advance() != kvmem::MemoryPhase::Ready) {}
        session.publish();
    }
    ensure_sequence_kv_mapped(sequence, end, backend_kv_pages ? end : 0);
    const auto mapped = kv_pages_for_tokens(compact_position(sequence, end));
    while (sequence.window.pages.size() < mapped) {
        const auto slot = static_cast<std::uint32_t>(sequence.window.pages.size());
        sequence.window.pages.push_back(slot + sequence.window.removed_tokens / kvmem_page_tokens);
    }
    if (text_kv_pages->physical_pool().allocated_pages() >
        static_cast<std::uint64_t>(max_concurrency) * kvmem_window_tokens / kvmem_page_tokens) {
        throw std::logic_error("KVMem exceeded its native physical page budget");
    }
}

// Capture uses the same complete-before-publish Host spill as a working-set
// transfer, but stops before any destructive restore. Native device ownership
// and attention view remain unchanged.
void ProgramImplCore::archive_memory_window(SequenceState& sequence) {
    WindowMemoryBackend backend(*this, sequence, sequence.text_kv_valid);
    const auto snapshot = backend.snapshot();
    kvmem::WorkingSetPlan plan;
    plan.expected = snapshot.stamp;
    for (auto id : snapshot.resident) {
        const auto& block = snapshot.blocks.at(static_cast<std::size_t>(id));
        plan.next.blocks.push_back({block.block, plan.next.visible_tokens});
        plan.next.visible_tokens += block.block.valid_tokens;
    }
    WindowMemoryTransfer spill(backend, plan, false);
    try {
        spill.start_spill();
    } catch (...) {
        (void)spill.abort();
        throw;
    }
}

bool ProgramImplCore::memory_query_probe(const SequenceState& sequence,
                                         std::uint32_t prompt_tokens) const {
    const auto& window = sequence.window;
    return kvmem_window_tokens && prompt_tokens > kvmem_window_tokens &&
        window.query_begin > 0 && window.query_end > window.query_begin && !window.query_replayed;
}

std::uint32_t ProgramImplCore::memory_checkpoint_frontier(const SequenceState& sequence,
                                                          std::uint32_t prompt_tokens,
                                                          bool allow_reuse) const {
    if (memory_query_probe(sequence, prompt_tokens)) return sequence.window.query_begin;
    // Leave one suffix token to evaluate and sample normally, including the MTP
    // boundary row rebuilt on restore. No query probe/replay is needed in this path.
    if (kvmem_window_tokens && allow_reuse && prompt_tokens > 1 &&
        prompt_tokens <= kvmem_window_tokens) {
        return prompt_tokens - 1;
    }
    return 0;
}

void ProgramImplCore::capture_memory_prefix(SequenceState& sequence,
                                            std::uint32_t chunk_tokens) {
    auto& window = sequence.window;
    if (!chunk_tokens || (window.prefix_checkpoint &&
        window.prefix_checkpoint->prefix.size() >= sequence.text_kv_valid)) {
        throw std::logic_error(
            "KVMem prefix checkpoint is not at an uncaptured evaluated boundary");
    }
    copy_tail(sequence,
              prefill_hidden.slice(1, static_cast<std::int32_t>(chunk_tokens) - 1, 1));
    window.prefix_checkpoint = capture_memory_checkpoint(sequence);
}

std::unique_ptr<KvmemPrefixCheckpoint> ProgramImplCore::capture_memory_checkpoint(
    SequenceState& sequence) {
    auto& window = sequence.window;
    auto checkpoint = std::make_unique<KvmemPrefixCheckpoint>();
    const auto frontier = sequence.text_kv_valid;
    if (!frontier || frontier > sequence.ledger.size() ||
        frontier != window.statistics_frontier || !sequence.tail_hidden_valid ||
        sequence.state.fork_pending ||
        (backend_kv_pages && sequence.mtp_kv_valid != frontier)) {
        throw std::logic_error("KVMem checkpoint is not a complete committed boundary");
    }
    checkpoint->prefix.assign(sequence.ledger.begin(), sequence.ledger.begin() + frontier);
    checkpoint->prefix_identity = sequence.prefix_identity;
    checkpoint->prefix_identity.truncate(frontier);
    checkpoint->mean_tail = window.statistics->mean_checkpoint(frontier);
    const auto bytes = state_images->host_layout().image_bytes;
    checkpoint->state = std::make_unique<PinnedHostBuffer>(bytes);
    std::memset(checkpoint->state->data(), 0, bytes);
    archive_memory_window(sequence);
    WindowMemoryBackend backend(*this, sequence, frontier);
    const auto descriptor = backend.descriptor();
    checkpoint->identity = {descriptor.identity, descriptor.layout, backend.stamp()};
    checkpoint->resident_pages = backend.snapshot().resident;
    state_images->copy_to_host(state_store->physical_slot(sequence.state.write),
        qwen3_6::HostStateImageView{static_cast<std::byte*>(checkpoint->state->data()),
                                    &state_images->host_layout()},
        device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    if (kvmem_options.verify_transfers) {
        std::fprintf(stderr, "KVMEM_PREFIX_CHECKPOINT frontier=%u state_bytes=%zu\n", frontier,
                     bytes);
    }
    return checkpoint;
}

std::unique_ptr<KvmemPrefixCheckpoint>& ProgramImplCore::memory_checkpoint_slot(
    KvmemWindowState& window, KvmemCheckpointKind kind) {
    switch (kind) {
    case KvmemCheckpointKind::Base: return window.prefix_checkpoint;
    case KvmemCheckpointKind::Endpoint: return window.endpoint_checkpoint;
    case KvmemCheckpointKind::InputFallback: return window.input_checkpoint;
    case KvmemCheckpointKind::Reconstruction: return window.reconstruction_checkpoint;
    }
    throw std::logic_error("invalid KVMem checkpoint kind");
}

const KvmemPrefixCheckpoint* ProgramImplCore::memory_checkpoint(const KvmemWindowState& window,
                                                               KvmemCheckpointKind kind) const {
    switch (kind) {
    case KvmemCheckpointKind::Base: return window.prefix_checkpoint.get();
    case KvmemCheckpointKind::Endpoint: return window.endpoint_checkpoint.get();
    case KvmemCheckpointKind::InputFallback: return window.input_checkpoint.get();
    case KvmemCheckpointKind::Reconstruction: return window.reconstruction_checkpoint.get();
    }
    return nullptr;
}

void ProgramImplCore::capture_memory_endpoint(SequenceState& sequence,
                                              const RequestControl& request) noexcept {
    if (!kvmem_window_tokens || !request.allow_memory_reuse || !sequence.window.statistics ||
        request.lifecycle != Lifecycle::Finishable) {
        return;
    }
    const auto frontier = sequence.execution_frontier;
    if (sequence.text_kv_valid != frontier || !sequence.tail_hidden_valid ||
        sequence.state.fork_pending || (backend_kv_pages && sequence.mtp_kv_valid != frontier)) {
        return;
    }
    for (const auto kind : kKvmemCheckpointKinds)
        if (const auto* checkpoint = memory_checkpoint(sequence.window, kind);
            checkpoint && checkpoint->prefix.size() == frontier) {
            return;
        }
    try {
        sequence.window.endpoint_checkpoint = capture_memory_checkpoint(sequence);
        if (kvmem_options.verify_transfers) {
            std::fprintf(stderr, "KVMEM_ENDPOINT frontier=%u committed=1\n", frontier);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVMEM_ENDPOINT capture skipped: %s\n", error.what());
    }
}

bool ProgramImplCore::memory_same_query(const KvmemWindowState& window,
                                        const PreparedPromptData& prompt,
                                        std::uint32_t frontier) const {
    if (!prompt.memory_query || !window.query_tokens ||
        window.query_tokens != window.query_end - window.query_begin ||
        frontier < window.query_end) {
        return false;
    }
    const auto& query = *prompt.memory_query;
    // The caller's exact checkpoint match proves all tokens/positions through this range.
    return query.begin == window.query_begin && query.count == window.query_tokens;
}

void ProgramImplCore::capture_memory_rewrite(SequenceState& sequence,
                                             KvmemCheckpointKind kind) noexcept {
    auto& window = sequence.window;
    const auto* purpose = kind == KvmemCheckpointKind::InputFallback ? "input" : "reconstruction";
    try {
        if (kind != KvmemCheckpointKind::InputFallback &&
            kind != KvmemCheckpointKind::Reconstruction) {
            throw std::logic_error("invalid KVMem rewrite checkpoint purpose");
        }
        if (!window.prefix_checkpoint ||
            window.prefix_checkpoint->prefix.size() != sequence.text_kv_valid) {
            memory_checkpoint_slot(window, kind) = capture_memory_checkpoint(sequence);
        }
        if (kvmem_options.verify_transfers) {
            std::fprintf(stderr, "KVMEM_REWRITE purpose=%s frontier=%u canonical=1\n", purpose,
                         sequence.text_kv_valid);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVMEM_REWRITE purpose=%s capture skipped: %s\n", purpose,
                     error.what());
    }
    if (kind == KvmemCheckpointKind::Reconstruction) window.reconstruction_frontier = 0;
}

void ProgramImplCore::capture_memory_reconstruction_if_ready(SequenceState& sequence) {
    auto& window = sequence.window;
    if (window.reconstruction_frontier < sequence.text_kv_valid) {
        window.reconstruction_frontier = 0;
    }
    if (!kvmem_window_tokens || !requests[sequence.lane].allow_memory_reuse ||
        !window.reconstruction_frontier ||
        window.reconstruction_frontier != sequence.text_kv_valid ||
        sequence.execution_frontier != sequence.text_kv_valid ||
        (backend_kv_pages && sequence.mtp_kv_valid != sequence.text_kv_valid)) {
        return;
    }
    capture_memory_rewrite(sequence, KvmemCheckpointKind::Reconstruction);
}

void ProgramImplCore::rewind_memory_query(SequenceState& sequence) {
    auto& window = sequence.window;
    if (!window.prefix_checkpoint || window.query_replayed ||
        window.query_tokens != window.query_end - window.query_begin ||
        sequence.state.read != sequence.state.write || sequence.state.fork_pending) {
        throw std::logic_error("KVMem query replay has no complete exclusive checkpoint");
    }
    const auto& checkpoint = *window.prefix_checkpoint;
    if (checkpoint.prefix.size() != window.query_begin) {
        throw std::logic_error("KVMem query checkpoint is not the current query boundary");
    }
    const auto previous_frontier = sequence.text_kv_valid;
    window.query_replayed = true; // Probe Q stays frozen while replay replaces K once.
    restore_memory_prefix(sequence, checkpoint);
    if (kvmem_options.verify_transfers) {
        std::fprintf(stderr,
            "KVMEM_QUERY_REPLAY from=%u to=%u query_tokens=%u prefix_match=1 state_exact=1\n",
            previous_frontier, sequence.text_kv_valid, window.query_tokens);
    }
}

void ProgramImplCore::restore_memory_prefix(SequenceState& sequence,
                                            const KvmemPrefixCheckpoint& checkpoint,
                                            bool preserve_view) {
    auto& window = sequence.window;
    const auto frontier = static_cast<std::uint32_t>(checkpoint.prefix.size());
    WindowMemoryBackend backend(*this, sequence, sequence.text_kv_valid);
    const auto descriptor = backend.descriptor();
    const kvmem::CheckpointIdentity expected{descriptor.identity, descriptor.layout,
                                             checkpoint.identity.stamp};
    if (!checkpoint.identity.compatible_with(expected)) {
        throw std::logic_error("KVMem stale checkpoint: identity/layout mismatch");
    }
    if (!(checkpoint.identity.stamp.session == window.stamp.session)) {
        throw std::logic_error("KVMem stale checkpoint: session mismatch");
    }
    if (frontier != checkpoint.identity.stamp.frontier) {
        throw std::logic_error("KVMem stale checkpoint: frontier vs stamp");
    }
    if (frontier > sequence.ledger.size()) {
        throw std::logic_error("KVMem stale checkpoint: frontier beyond ledger");
    }
    if (!std::equal(checkpoint.prefix.begin(), checkpoint.prefix.end(), sequence.ledger.begin())) {
        throw std::logic_error("KVMem stale checkpoint: token prefix mismatch");
    }
    if (!checkpoint.prefix_identity.prefix_equals(sequence.prefix_identity, frontier)) {
        throw std::logic_error("KVMem stale checkpoint: prefix identity mismatch");
    }
    const auto bytes = state_images->host_layout().image_bytes;
    // All prefix records were made durable at capture. Query appends cannot alter
    // native values/scales in earlier rows; the partial page is explicitly clipped.
    const auto pages = kv_pages_for_tokens(frontier);
    if (window.archive.size() < pages) {
        throw std::logic_error("KVMem checkpoint lost Host prefix");
    }
    for (std::uint32_t id = 0; id < pages; ++id) {
        if (!window.archive[id] ||
            window.host_versions[id] < kvmem_valid_columns(id, frontier)) {
            throw std::logic_error("KVMem checkpoint has an incomplete native prefix");
        }
    }
    window.statistics->truncate_to(frontier);
    window.statistics->restore_mean_checkpoint(frontier, checkpoint.mean_tail);
    window.selector->truncate_to(frontier);
    window.statistics_frontier = frontier;
    window.archive.resize(pages);
    window.host_versions.resize(pages);
    window.host_bytes = pages * backend.page_bytes;
    if (frontier % kvmem_page_tokens) {
        window.host_versions.back() = frontier % kvmem_page_tokens;
    }
    // A consumed history has one archive. Rewinding explicitly revokes later states
    // before their payload/statistics can be overwritten; no full-history COW is needed.
    for (const auto kind : kKvmemCheckpointKinds) {
        auto& slot = memory_checkpoint_slot(window, kind);
        if (slot && slot->prefix.size() > frontier) slot.reset();
    }
    sequence.text_kv_valid = frontier;
    if (backend_kv_pages) {
        sequence.mtp_kv_valid = frontier;
        sequence.mtp_draft_count = 0;
    }
    ++window.stamp.execution_history;
    state_store->begin_active_overwrite(sequence.state.write);
    // The new Root StateImage can still have reset work on the execution
    // stream. Complete it before restoring bytes on the transfer stream.
    device.synchronize();
    state_images->copy_from_host(
        qwen3_6::HostStateImageConstView{static_cast<const std::byte*>(checkpoint.state->data()),
                                         &state_images->host_layout()},
        state_store->physical_slot(sequence.state.write), device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    if (kvmem_options.verify_transfers) {
        PinnedHostBuffer verify(bytes);
        std::memset(verify.data(), 0, bytes);
        state_images->copy_to_host(state_store->physical_slot(sequence.state.write),
            qwen3_6::HostStateImageView{static_cast<std::byte*>(verify.data()),
                                        &state_images->host_layout()},
            device.transfer_stream);
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
        if (std::memcmp(verify.data(), checkpoint.state->data(), bytes)) {
            throw std::logic_error("KVMem query StateImage restore changed native bytes");
        }
    }
    if (preserve_view) {
        auto restored = std::make_shared<WindowMemoryBackend>(*this, sequence, frontier);
        kvmem::MemorySession session(restored);
        // A checkpoint includes already appended reserve rows. Restore its exact
        // view within B+R, then ordinary append pressure restores the B/R split.
        session.prepare(restored->stamp(), checkpoint.resident_pages, checkpoint.resident_pages,
            {kvmem_window_tokens, 0, kvmem_window_tokens, kvmem_options.host_bytes});
        while (session.advance() != kvmem::MemoryPhase::Ready) {}
        session.publish();
    } else {
        prepare_window(sequence, frontier, frontier, true);
    }
    sequence.tail_hidden_valid = true;
}

void ProgramImplCore::bind_memory_snapshot(SequenceState& sequence) {
    WindowMemoryBackend backend(*this, sequence, sequence.text_kv_valid);
    const auto descriptor = backend.descriptor();
    for (const auto kind : kKvmemCheckpointKinds) {
        auto* checkpoint = memory_checkpoint_slot(sequence.window, kind).get();
        if (!checkpoint) continue;
        auto stamp = backend.stamp();
        stamp.frontier = static_cast<std::uint32_t>(checkpoint->prefix.size());
        checkpoint->identity = {descriptor.identity, descriptor.layout, stamp};
    }
}

// ---- Retained Host histories --------------------------------------------------------------

std::optional<KvmemHistoryMatch> ProgramImplCore::memory_restorable_history(
    const PreparedPromptData& prompt) const {
    if (!kvmem_window_tokens || !prompt.identity.reusable ||
        (prompt.context_cache.session_key && !prompt.context_cache.update_session_index)) {
        return std::nullopt;
    }
    std::optional<KvmemHistoryMatch> best;
    std::size_t best_frontier = 0;
    unsigned prefix_rejected = 0, identity_rejected = 0, query_rejected = 0, incomplete = 0;
    for (std::size_t index = 0; index < memory_histories.size(); ++index) {
        const auto& history = memory_histories[index];
        if (!history || !history->stamp.valid ||
            history->session_key != prompt.context_cache.session_key) {
            continue;
        }
        for (const auto kind : kKvmemCheckpointKinds) {
            // The Base prefix checkpoint (a short prompt's pre-final-token capture) is restored
            // through the MTP BeforeSuffix bridge flow, which this port does not carry yet; only
            // complete execution endpoints participate in Host-history reuse.
            if (kind == KvmemCheckpointKind::Base) { continue; }
            const auto* candidate = memory_checkpoint(*history, kind);
            if (!candidate) continue;
            const auto& checkpoint = *candidate;
            const auto frontier = static_cast<std::uint32_t>(checkpoint.prefix.size());
            const auto pages = kv_pages_for_tokens(frontier);
            bool complete = frontier && checkpoint.state &&
                checkpoint.state->size() == state_images->host_layout().image_bytes &&
                frontier == checkpoint.identity.stamp.frontier && history->statistics &&
                history->selector && history->statistics_frontier == frontier &&
                history->archive.size() >= pages && history->host_versions.size() >= pages;
            for (std::uint32_t page = 0; complete && page < pages; ++page) {
                complete = history->archive[page] &&
                           history->host_versions[page] >= kvmem_valid_columns(page, frontier);
            }
            if (!complete) { ++incomplete; continue; }
            if (frontier >= prompt.token_ids.size() ||
                !std::equal(checkpoint.prefix.begin(), checkpoint.prefix.end(),
                            prompt.token_ids.begin())) {
                ++prefix_rejected;
                continue;
            }
            if (!checkpoint.prefix_identity.matches(prompt, frontier)) {
                ++identity_rejected;
                continue;
            }
            if (prompt.token_ids.size() > kvmem_window_tokens && prompt.memory_query &&
                frontier > prompt.memory_query->begin &&
                !memory_same_query(*history, prompt, frontier)) {
                ++query_rejected;
                continue;
            }
            if (frontier > best_frontier ||
                (frontier == best_frontier && best &&
                 history->publication_order >
                     memory_histories[best->history]->publication_order)) {
                best = KvmemHistoryMatch{index, kind, frontier};
                best_frontier = frontier;
            }
        }
    }
    if (kvmem_options.verify_transfers) {
        std::fprintf(stderr,
            "KVMEM_HISTORY_LOOKUP frontier=%zu prefix_rejected=%u identity_rejected=%u "
            "query_rejected=%u incomplete=%u\n",
            best_frontier, prefix_rejected, identity_rejected, query_rejected, incomplete);
    }
    return best;
}

bool ProgramImplCore::restore_memory_history(SequenceState& sequence,
                                             RequestControl::Prefill& staged) {
    const auto match = memory_restorable_history(staged.prompt);
    if (!match) return false;
    auto& history = memory_histories[match->history];
    const auto frontier = std::optional<std::uint32_t>(match->frontier);
    if (*frontier != staged.memory_restore_frontier ||
        staged.memory_restore_generation != history->stamp.session.id) {
        return false;
    }
    sequence.window = std::move(*history);
    history.reset();
    auto& window = sequence.window;
    const bool same_query = memory_same_query(window, staged.prompt, *frontier);
    auto checkpoint = std::move(memory_checkpoint_slot(window, match->kind));
    // The previous Program address-space lease has been released. These IDs
    // must never be treated as resident in the new native address space.
    window.pages.clear();
    window.removed_tokens = 0;
    window.reconstruction_frontier = 0;
    window.query_replayed = same_query;
    const auto query = staged.prompt.memory_query.value_or(TokenSpan{});
    window.query_begin = static_cast<std::uint32_t>(query.begin);
    window.query_end = static_cast<std::uint32_t>(query.begin + query.count);
    if (!same_query) {
        window.query_tokens = 0;
        std::fill(window.query_sum.begin(), window.query_sum.end(), 0.0F);
    }
    // The durable archive already covers the quoted prefix. This is not a
    // native allocation frontier; force-selection creates only B resident rows.
    sequence.text_kv_valid = *frontier;
    if (backend_kv_pages) sequence.mtp_kv_valid = *frontier;
    restore_memory_prefix(sequence, *checkpoint, true);
    if (backend_kv_pages) {
        // An MTP row combines target hidden at i with the embedding of token i+1.
        // Prefix identity proves only [0,frontier), so its last MTP row must be
        // rebuilt using this request's first suffix token before it is trusted.
        const auto last = *frontier - 1U;
        sequence.mtp_kv_valid = last;
        trim_sequence_kv(sequence, *frontier, last);
        ensure_sequence_kv_mapped(sequence, *frontier, *frontier);
        const auto selectors = state_selectors(sequence);
        schedule::PrefillContext bridge{
            {device, model, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
            text_kv_view(sequence),
            mtp_kv_view(sequence),
            decoder->text_kv,
            decoder->mtp_cache(),
            nullptr,
            *frontier,
            nullptr,
            nullptr,
            selectors.source,
            selectors.destination,
            0,
            nullptr};
        bridge.cache_position_shift = window.removed_tokens;
        set_device_i32(io.rope_delta,
                       sequence.rope_delta + checked_i32(window.removed_tokens,
                                                         "restored MTP cache shift"));
        auto token = io.mtp->target_input_ids.slice(0, 0, 1);
        set_device_i32(token, staged.prompt.token_ids[*frontier]);
        const auto rope = prompt_rope_position(staged.prompt, last);
        mark_workspace_usage(workspace_plan.mtp_prefill);
        if (staged.vision) {
            schedule::mtp_bridge_multimodal(bridge, staged.prompt, *staged.vision,
                {.previous_hidden = &sequence.tail_hidden,
                 .position = checked_i32(last, "restored MTP tail"),
                 .rope_position = rope});
        } else {
            schedule::mtp_bridge_and_propose(bridge, token, sequence.tail_hidden,
                                             checked_i32(last, "restored MTP tail"), rope, false);
        }
        device.synchronize();
        work.reset();
        sequence.mtp_kv_valid = *frontier;
        commit_sequence_kv(sequence, *frontier, *frontier);
        window.host_versions.at(last / kvmem_page_tokens) = 0;
        // Same-boundary query replay can restore before another view switch.
        archive_memory_window(sequence);
    }
    if (*frontier == memory_checkpoint_frontier(sequence, staged.prompt_tokens, true)) {
        window.prefix_checkpoint = std::move(checkpoint);
    } else {
        // An exact replay can start at the input capture boundary, so prefill
        // will not cross it again. Keep the restored complete state for repeats
        // and cancellation recovery; restoring does not consume its payload.
        memory_checkpoint_slot(window, match->kind) = std::move(checkpoint);
    }
    staged.base = *frontier;
    staged.cursor = *frontier;
    ++memory_history_hits;
    if (kvmem_options.verify_transfers) {
        std::fprintf(stderr,
            "KVMEM_HISTORY_RESTORE frontier=%u kind=%u same_query=%u prefix_match=1 "
            "state_exact=1\n",
            *frontier, static_cast<unsigned>(match->kind), same_query ? 1U : 0U);
    }
    return true;
}

void ProgramImplCore::reserve_memory_history(SequenceState& sequence, RequestControl& request,
                                             const PreparedPromptData& prompt) {
    const auto& key = prompt.context_cache.session_key;
    if (key && prompt.context_cache.update_session_index) {
        MemoryPublication* target = nullptr;
        for (auto& publication : memory_publications) {
            if (publication.key == key) { target = &publication; break; }
            if (!publication.key) { target = &publication; }
        }
        if (!target) {
            for (auto& publication : memory_publications) {
                const bool active = std::any_of(requests.begin(), requests.end(),
                    [&](const auto& value) { return value.memory_session_key == publication.key; });
                const bool retained = std::any_of(memory_histories.begin(), memory_histories.end(),
                    [&](const auto& value) {
                        return value && value->session_key == publication.key;
                    });
                if (!active && !retained && (!target || publication.order < target->order)) {
                    target = &publication;
                }
            }
        }
        if (!target) throw std::logic_error("KVMem publication table has no reclaimable entry");
        if (target->key != key) { target->key = key; target->order = 0; }
        target->order = std::max(target->order, sequence.window.publication_order);
    }
    if (!sequence.window.statistics) ++memory_history_misses;
    // An explicit fresh request invalidates only its own lineage. A caller that
    // forbids index updates cannot consume or erase the named cached history.
    if (!request.allow_memory_reuse && prompt.context_cache.update_session_index) {
        if (prompt.context_cache.session_key) erase_memory_snapshot(*prompt.context_cache.session_key);
        for (auto& history : memory_histories) {
            if (history && history->session_key == prompt.context_cache.session_key &&
                history->publication_order <= sequence.window.publication_order) {
                history.reset();
            }
        }
    }
    std::uint64_t reserved = 0;
    for (const auto& active : requests) reserved += active.memory_host_reservation;
    if (reserved > kvmem_options.host_bytes) {
        throw std::logic_error("KVMem active Host reservations exceed admission capacity");
    }
    auto available = kvmem_options.host_bytes - reserved;
    for (;;) {
        std::uint64_t cached_bytes = 0;
        std::optional<std::size_t> victim;
        for (std::size_t index = 0; index < memory_histories.size(); ++index) {
            const auto& history = memory_histories[index];
            if (!history) continue;
            cached_bytes += history->host_bytes;
            if (!victim ||
                history->publication_order < memory_histories[*victim]->publication_order) {
                victim = index;
            }
        }
        if (cached_bytes <= available) break;
        if (!victim) throw std::logic_error("KVMem retained Host accounting has no victim");
        memory_histories[*victim].reset();
        ++memory_history_evictions;
    }
}

void ProgramImplCore::retain_memory_history(SequenceState& sequence,
                                            const RequestControl& request) noexcept {
    if (!kvmem_window_tokens || !request.allow_memory_reuse || !sequence.window.stamp.valid ||
        (!sequence.window.prefix_checkpoint && !sequence.window.endpoint_checkpoint &&
         !sequence.window.input_checkpoint && !sequence.window.reconstruction_checkpoint)) {
        return;
    }
    auto& window = sequence.window;
    if (window.session_key && !window.update_session_index) return;
    if (window.session_key) {
        for (const auto& publication : memory_publications) {
            if (publication.key == window.session_key &&
                publication.order > window.publication_order) {
                return;
            }
        }
    }
    std::uint32_t frontier = 0;
    for (const auto kind : kKvmemCheckpointKinds)
        if (const auto* checkpoint = memory_checkpoint(window, kind))
            frontier = std::max(frontier, static_cast<std::uint32_t>(checkpoint->prefix.size()));
    const auto pages = kv_pages_for_tokens(frontier);
    if (!pages || pages > window.archive.size() || pages > window.host_versions.size() ||
        !window.archive.front()) {
        return;
    }
    const auto page_bytes = window.archive.front()->size();
    // All slots share one archive through the latest complete checkpoint.
    // Uncommitted/cancelled rows past that frontier are never retained.
    window.archive.resize(pages);
    window.host_versions.resize(pages);
    window.host_bytes = pages * page_bytes;
    std::optional<std::size_t> destination;
    std::optional<std::size_t> oldest;
    std::size_t count = 0;
    for (std::size_t index = 0; index < memory_histories.size(); ++index) {
        auto& history = memory_histories[index];
        if (!history) { if (!destination) destination = index; continue; }
        ++count;
        if (window.session_key && history->session_key == window.session_key) {
            if (history->publication_order > window.publication_order) return;
            destination = index;
            break;
        }
        if (!oldest ||
            history->publication_order < memory_histories[*oldest]->publication_order) {
            oldest = index;
        }
    }
    if (count >= kvmem_options.retained_sessions &&
        (!destination || !memory_histories[*destination])) {
        destination = oldest;
        ++memory_history_evictions;
    }
    if (!destination) return;
    // Capture already completed every Host copy. The active state's later
    // writes and unpublished output never become this checkpoint's contents.
    save_memory_snapshot(window);
    memory_histories[*destination].emplace(std::move(window));
}

// Cold disk snapshots are not enabled in this build (kvmem.disk_path/disk_bytes are rejected at
// startup); the in-process retained histories above are the whole persistence surface.
void ProgramImplCore::save_memory_snapshot(KvmemWindowState&) noexcept {}
void ProgramImplCore::load_memory_snapshot(const PreparedPromptData&) noexcept {}
void ProgramImplCore::erase_memory_snapshot(const PreparedSessionKey&) noexcept {}

// ---- Mean-K statistics (port of program/storage/memory_statistics.cpp) -------------------

namespace {
void read_memory_statistics(DeviceContext& device,
                            const qwen3_6::detail::MemoryStatistics& statistics,
                            void* destination, bool include_query) {
    auto* host = static_cast<std::byte*>(destination);
    const auto key_stride = std::size_t(statistics.key_width) * statistics.columns * sizeof(float);
    const auto query_stride = std::size_t(statistics.query_width) * sizeof(float);
    for (std::size_t rank = 0; rank < statistics.ranks.size(); ++rank) {
        const auto& shard = statistics.ranks[rank];
        if (shard.key_sums.data == nullptr) { continue; }
        CUDA_CHECK(cudaMemcpyAsync(host + shard.first_layer * key_stride, shard.key_sums.data,
                                   shard.key_sums.bytes(), cudaMemcpyDeviceToHost, device.stream));
        if (include_query) {
            CUDA_CHECK(cudaMemcpyAsync(
                host + statistics.key_bytes() + shard.first_layer * query_stride,
                shard.query_sums.data, shard.query_sums.bytes(), cudaMemcpyDeviceToHost,
                device.stream));
        }
    }
    device.synchronize();
}
} // namespace

void ProgramImplCore::prepare_memory_statistics(SequenceState& sequence,
                                                std::uint32_t query_begin,
                                                std::uint32_t query_end) {
    if (!memory_statistics) { return; }
    auto& state = sequence.window;
    if (!state.statistics) {
        state.statistics = std::make_unique<kvmem::RawKvStore>(kvmem::RawKvStoreConfig{
            .n_layer = static_cast<std::uint32_t>(TextConfig::full_attention_layers()),
            .n_embd_k = memory_statistics->key_width,
            .block_tokens = 64});
        kvmem::KvMemStoreConfig selection;
        selection.block_tokens = 64;
        selection.select_budget = kvmem_options.selected_tokens;
        selection.sink_blocks = kvmem_options.sink_blocks();
        selection.recent_blocks = kvmem_options.recent_blocks();
        state.selector = std::make_unique<kvmem::KvMemStore>(selection);
        state.query_sum.assign(memory_statistics->query_elements(), 0.0F);
        state.query_begin = query_begin;
        state.query_end = query_end;
    }
    if (state.query_begin != query_begin || state.query_end != query_end) {
        throw std::logic_error("KVMem query range changed inside one prefill");
    }
    memory_statistics_ranges = {0, static_cast<std::int32_t>(capacity),
        static_cast<std::int32_t>(query_begin), static_cast<std::int32_t>(query_end)};
    for (std::size_t rank = 0; rank < memory_statistics->ranks.size(); ++rank) {
        const auto& shard = memory_statistics->ranks[rank];
        if (shard.ranges.data == nullptr) { continue; }
        CUDA_CHECK(cudaMemcpyAsync(shard.ranges.data, memory_statistics_ranges.data(),
                                   sizeof(memory_statistics_ranges), cudaMemcpyHostToDevice,
                                   device.stream));
    }
}

void ProgramImplCore::commit_memory_statistics(SequenceState& sequence, std::uint32_t begin,
                                               std::uint32_t end) {
    if (!memory_statistics) { return; }
    auto& state = sequence.window;
    if (!state.statistics || begin != state.statistics_frontier || end <= begin ||
        end - begin > prefill_chunk) {
        throw std::logic_error(
            "KVMem statistics require exactly one append at the evaluated frontier");
    }
    const auto& buffers = *memory_statistics;
    auto* host = static_cast<std::byte*>(memory_statistics_host->data());
    read_memory_statistics(device, buffers, host, true);
    const auto width = buffers.key_width;
    const auto layer_stride = static_cast<std::size_t>(width) * buffers.columns;
    const auto* keys = reinterpret_cast<const float*>(host);
    for (std::uint32_t layer = 0; layer < state.statistics->config().n_layer; ++layer) {
        for (auto at = begin; at < end;) {
            const auto count = std::min(end - at, kvmem_page_tokens - at % kvmem_page_tokens);
            const auto bucket = at / kvmem_page_tokens - begin / kvmem_page_tokens;
            const auto* sum = keys + layer * layer_stride + bucket * width;
            for (std::uint32_t channel = 0; channel < width; ++channel) {
                if (!std::isfinite(sum[channel])) {
                    throw std::runtime_error("Nonfinite KVMem key sum");
                }
            }
            state.statistics->write_layer_mean_sum(at, count, layer, sum);
            at += count;
        }
    }
    const auto query_first = std::max(begin, state.query_begin);
    const auto query_last = std::min(end, state.query_end);
    if (!state.query_replayed && query_last > query_first) {
        const auto* query = reinterpret_cast<const float*>(host + buffers.key_bytes());
        for (std::size_t i = 0; i < state.query_sum.size(); ++i) {
            if (!std::isfinite(query[i])) {
                throw std::runtime_error("Nonfinite KVMem query sum");
            }
            state.query_sum[i] += query[i];
        }
        state.query_tokens += query_last - query_first;
    }
    if (state.selector->total_tokens() != begin) {
        throw std::logic_error("KVMem block catalog lost its statistics frontier");
    }
    state.selector->register_append(end - begin);
    state.statistics_frontier = end;
    if (kvmem_options.verify_transfers &&
        (end == state.query_end || end % kvmem_page_tokens == 0)) {
        std::fprintf(stderr,
            "KVMEM_STATISTICS frontier=%u query_tokens=%u query_begin=%u query_end=%u layers=%u "
            "mean_bytes=%zu\n",
            end, state.query_tokens, state.query_begin, state.query_end,
            state.statistics->config().n_layer, state.statistics->allocated_bytes());
    }
}

void ProgramImplCore::read_memory_candidates() {
    if (!memory_candidate_statistics) return;
    read_memory_statistics(device, *memory_candidate_statistics, memory_candidate_host->data(),
                           false);
}

void ProgramImplCore::commit_memory_candidates(SequenceState& sequence, std::uint32_t begin,
                                               std::uint32_t end, std::uint32_t first_column,
                                               std::uint32_t row_width) {
    if (!memory_candidate_statistics) { return; }
    auto& state = sequence.window;
    const auto& keys = *memory_candidate_statistics;
    if (!state.statistics || !state.selector || begin != state.statistics_frontier ||
        state.selector->total_tokens() != begin || end <= begin || end - begin > row_width ||
        first_column + row_width > keys.columns) {
        throw std::logic_error(
            "KVMem speculative statistics have an invalid committed prefix");
    }
    const auto width = static_cast<std::size_t>(keys.key_width);
    const auto stride = width * keys.columns;
    const auto* values = static_cast<const float*>(memory_candidate_host->data());
    // Each represented BF16 key was copied into its own FP32 column by
    // token_sums(block_tokens=1). Rejected and output-truncated columns never
    // enter the persistent mean-K store, including a partial logical page.
    for (std::uint32_t layer = 0; layer < state.statistics->config().n_layer; ++layer) {
        for (auto position = begin; position < end; ++position) {
            const auto* value = values + layer * stride + (first_column + position - begin) * width;
            for (std::size_t channel = 0; channel < width; ++channel) {
                if (!std::isfinite(value[channel])) {
                    throw std::runtime_error("Nonfinite committed KVMem candidate key");
                }
            }
            state.statistics->write_layer_mean_sum(position, 1, layer, value);
        }
    }
    state.selector->register_append(end - begin);
    state.statistics_frontier = end;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
