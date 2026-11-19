#pragma once

#include "server-common.h"
#include "server-task.h"

#include <list>
#include <memory>
#include <vector>

// which prompt goes to which slot: slot prompt state, prompt cache and slot selection

// what a cached prompt can give to a new prompt
struct server_prompt_reuse {
    int32_t   n_past   = 0; // tokens to keep
    llama_pos pos_next = 0;

    // restore this checkpoint first, nullptr when the live state is used as it is
    // points into server_prompt::checkpoints, valid until that list changes
    const common_prompt_checkpoint * ckpt = nullptr;
};

struct server_prompt {
    server_tokens tokens;

    // by value: a device-resident checkpoint (dev_slot >= 0) belongs to its slot's storage pool,
    // so the prompt cache keeps host copies only (see server_prompt_cache::alloc)
    std::list<common_prompt_checkpoint> checkpoints;

    // the lora config that the tokens were processed with, it follows the state to the prompt cache and back
    // TODO: slot files do not store it
    std::vector<common_adapter_lora_info> lora;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
            lora,
        };
    }

    // n_prefix: tokens shared with the new prompt, must be > 0
    // has_new_tokens: the new prompt is longer than n_prefix
    // n_swa: 0 if a state must end right before the next token (no SWA, or recurrent)
    // pos_min: min position of the live state
    // n_past == 0 without a checkpoint means that nothing can be reused
    server_prompt_reuse get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const;
};

struct server_prompt_data {
    common_state_bytes main;
    common_state_bytes drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt.size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens);

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    // The largest state the cache restored from, kept for the next save: a few hundred MiB cost ~50 ms to free and
    //   ~100 ms of page faults to fill again, both while the request waits. One for the main and one for the draft state.
    common_state_bytes spare;
    common_state_bytes spare_drft;

    // A save of at most save_async_max bytes queues its copies (LLAMA_STATE_SEQ_FLAGS_ASYNC): one batch of
    //   device-to-pinned copies on the compute streams instead of a waited pageable copy per tensor, which ran at
    //   1.6-2 GB/s with the new entry's page faults. The pinned staging grows to the largest async save.
    //   env LLAMA_CACHE_SAVE_ASYNC: 0 = waited copies, 1 = queued copies waited for in the save,
    //   2 (default) = the bytes land in the entry when the next decode returns, its graphs queued behind the copies
    int    save_async     = 2;
    size_t save_async_max = 0;

    // contexts whose async save may still be copying into an entry: wait_saves() before an entry is freed
    std::vector<llama_context *> ctx_saving;

    void wait_saves();

    size_t size() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    // lora_new: the lora config of the new prompt, the states made with another config are skipped
    // replaced (optional): set to true if the slot's prompt was swapped for a cache entry
    bool load(server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, bool * replaced = nullptr);

    void update();
};

// what the scheduler needs to know about a slot
struct server_slot_info {
    int id = -1;
    const server_task * task = nullptr; // only for logging

    bool is_processing = false;
    int64_t t_last_used = -1;

    const server_prompt * prompt = nullptr;

    // prompt.tokens.size() when the slot's last generation began, -1 if unknown (see server_slot::n_prompt_gen_start)
    int64_t n_prompt_gen_start = -1;
};

struct server_slot_plan {
    // -1 if no slot is available; a busy slot if the task asks for it by id, the caller defers the task
    int id_slot = -1;

    // save the slot state to the prompt cache, then load the best match for the task
    bool update_cache = false;
};

// decides only, the caller applies the plan
struct server_slot_sched {
    float slot_prompt_similarity = 0.0f;

    // --slot-sim-use-cache: an LCP-similar slot that only partially overlaps also goes through the prompt cache
    bool slot_sim_use_cache = false;

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<int> get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const;

    // lora: the lora config of the task
    server_slot_plan select(const std::vector<server_slot_info> & slots, const server_task & task, const std::vector<common_adapter_lora_info> & lora) const;
};
