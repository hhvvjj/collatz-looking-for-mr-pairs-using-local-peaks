/*
######################################################################################################################
# looking_for_mr_pairs_using_local_peks.c
#
# Implementation to search mr values using local peak <= global peak approach 
#
# The complete article can be found on https://doi.org/10.5281/zenodo.15546925
#
#######################################################################################################################
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <omp.h>

/* ====================================================================================================================
 * CONSTANTS AND CONFIGURATION
 * ==================================================================================================================== */

#define MAX_SEQUENCE_LENGTH 50000
#define INITIAL_M_CAPACITY 100
#define HASH_TABLE_SIZE 8192
#define MAX_KNOWN_MR 100
#define PROGRESS_UPDATE_INTERVAL 3.0
#define PROGRESS_CHECK_FREQUENCY 100000
#define MIN_CHUNK_SIZE 100
#define MAX_CHUNK_SIZE 10000
#define SMALL_RANGE_THRESHOLD 1000000
#define MEDIUM_RANGE_THRESHOLD 100000000
#define MAX_DETAILED_DISCOVERIES 50

/* ====================================================================================================================
 * DATA STRUCTURES
 * ==================================================================================================================== */

// Core data types
typedef struct {
    uint64_t mr;
    uint64_t local_peak;
} MrLocalPeak;

typedef struct HashNode {
    uint64_t value;
    struct HashNode* next;
} HashNode;

typedef struct {
    HashNode* buckets[HASH_TABLE_SIZE];
    uint64_t* values;
    int count;
    int capacity;
} MValues;

// Result types
typedef struct {
    uint64_t mr;
    uint64_t first_n;
    uint64_t global_peak;
} MrResult;

typedef struct {
    uint64_t n;
    uint64_t mr;
    uint64_t local_peak;
    uint64_t global_peak;
} ViolationResult;

typedef struct {
    uint64_t mr;
    uint64_t first_n;
    uint64_t global_peak;
    bool is_known;
} UniqueDiscovery;

// Scheduling strategy
typedef enum {
    SCHEDULE_STATIC = 0,
    SCHEDULE_GUIDED = 1,
    SCHEDULE_DYNAMIC = 2
} SchedulingStrategy;

// Main containers
typedef struct {
    HashNode* seen_buckets[HASH_TABLE_SIZE];
    UniqueDiscovery* discoveries;
    int count;
    int capacity;
    omp_lock_t lock;
} DiscoveryTracker;

typedef struct {
    MrResult* known_results;
    MrResult* new_results;
    ViolationResult* violations;
    
    int known_count;
    int new_count;
    int violation_count;
    
    int known_capacity;
    int new_capacity;
    int violation_capacity;
    
    omp_lock_t lock;
} ValidationResults;

typedef struct {
    uint64_t processed;
    uint64_t found_count;
    uint64_t valid_mr_count;
    double last_progress_time;
    omp_lock_t lock;
} ProgressTracker;

typedef struct {
    SchedulingStrategy strategy;
    uint64_t chunk_size;
    int num_threads;
} SchedulingConfig;

typedef struct {
    const MrLocalPeak* known_dict;
    int dict_size;
    DiscoveryTracker* discovery;
    ValidationResults* results;
    ProgressTracker* progress;
} ValidationEngine;

/* ====================================================================================================================
 * KNOWN (MR,LOCAL_PEAK) DICTIONARY
 * ==================================================================================================================== */

static const MrLocalPeak KNOWN_MR_DICT[] = {
    {1, 7}, {2, 4}, {3, 25}, {6, 25}, {7, 79}, {8, 16}, {9, 43}, {12, 43}, {16, 49}, {19, 151},
    {25, 115}, {45, 4615}, {53, 106}, {60, 4615}, {79, 4615}, {91, 4615}, {121, 4615}, 
    {125, 889}, {141, 889}, {166, 889}, {188, 889}, {205, 2347}, {243, 4615}, {250, 889},
    {324, 4615}, {333, 10843}, {432, 4615}, {444, 10843}, {487, 4939}, {576, 4615},
    {592, 10843}, {649, 4615}, {667, 10843}, {683, 3643}, {865, 4615}, {889, 10843},
    {1153, 5191}, {1214, 13849}, {1821, 9223}, {2428, 9223}, {3643, 24595}
};

static const int KNOWN_MR_COUNT = sizeof(KNOWN_MR_DICT) / sizeof(KNOWN_MR_DICT[0]);

/* ====================================================================================================================
 * UTILITY FUNCTIONS
 * ==================================================================================================================== */

static inline uint64_t hash_function(uint64_t value) {
    return value & (HASH_TABLE_SIZE - 1);
}

static inline uint64_t calculate_m(uint64_t c) {
    uint64_t p = (c & 1) ? 1 : 2;
    return (c - p) >> 1;
}

static inline bool apply_collatz_transform(uint64_t* n) {
    if (*n & 1) {
        if (*n > (UINT64_MAX - 1) / 3) return false;
        uint64_t temp = 3 * (*n);
        if (temp > UINT64_MAX - 1) return false;
        *n = temp + 1;
    } else {
        *n = *n >> 1;
    }
    return true;
}

static void handle_memory_error(const char* operation) {
    fprintf(stderr, "Error: Memory %s failed\n", operation);
    exit(1);
}

static void* safe_malloc(size_t size, const char* context) {
    void* ptr = malloc(size);
    if (!ptr) handle_memory_error("allocation");
    return ptr;
}

static void* safe_realloc(void* ptr, size_t size, const char* context) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr) handle_memory_error("reallocation");
    return new_ptr;
}

/* ====================================================================================================================
 * KNOWN MR DICTIONARY OPERATIONS
 * ==================================================================================================================== */

static uint64_t find_known_mr_local_peak(uint64_t mr) {
    for (int i = 0; i < KNOWN_MR_COUNT; i++) {
        if (KNOWN_MR_DICT[i].mr == mr) {
            return KNOWN_MR_DICT[i].local_peak;
        }
    }
    return 0;
}

static bool is_known_mr(uint64_t mr) {
    return find_known_mr_local_peak(mr) > 0;
}

/* ====================================================================================================================
 * M VALUES CONTAINER IMPLEMENTATION
 * ==================================================================================================================== */

static void init_m_values(MValues* mv) {
    mv->capacity = INITIAL_M_CAPACITY;
    mv->values = safe_malloc(mv->capacity * sizeof(uint64_t), "m_values");
    mv->count = 0;
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        mv->buckets[i] = NULL;
    }
}

static void destroy_m_values(MValues* mv) {
    if (!mv) return;
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        HashNode* current = mv->buckets[i];
        while (current) {
            HashNode* next = current->next;
            free(current);
            current = next;
        }
        mv->buckets[i] = NULL;
    }
    
    free(mv->values);
    mv->values = NULL;
    mv->count = 0;
    mv->capacity = 0;
}

static bool is_m_repeated(const MValues* mv, uint64_t m) {
    uint64_t hash = hash_function(m);
    HashNode* current = mv->buckets[hash];
    
    while (current) {
        if (current->value == m) return true;
        current = current->next;
    }
    return false;
}

static void add_m_value(MValues* mv, uint64_t m) {
    if (mv->count >= mv->capacity) {
        mv->capacity *= 2;
        mv->values = safe_realloc(mv->values, mv->capacity * sizeof(uint64_t), "m_values expansion");
    }
    
    mv->values[mv->count++] = m;
    
    uint64_t hash = hash_function(m);
    HashNode* new_node = safe_malloc(sizeof(HashNode), "hash node");
    new_node->value = m;
    new_node->next = mv->buckets[hash];
    mv->buckets[hash] = new_node;
}

/* ====================================================================================================================
 * DISCOVERY TRACKER IMPLEMENTATION
 * ==================================================================================================================== */

static DiscoveryTracker* create_discovery_tracker(void) {
    DiscoveryTracker* tracker = safe_malloc(sizeof(DiscoveryTracker), "discovery tracker");
    
    tracker->capacity = 1000;
    tracker->discoveries = safe_malloc(tracker->capacity * sizeof(UniqueDiscovery), "discoveries");
    tracker->count = 0;
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        tracker->seen_buckets[i] = NULL;
    }
    
    omp_init_lock(&tracker->lock);
    return tracker;
}

static void destroy_discovery_tracker(DiscoveryTracker* tracker) {
    if (!tracker) return;
    
    free(tracker->discoveries);
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        HashNode* current = tracker->seen_buckets[i];
        while (current) {
            HashNode* next = current->next;
            free(current);
            current = next;
        }
    }
    
    omp_destroy_lock(&tracker->lock);
    free(tracker);
}

static bool has_mr_been_seen(const DiscoveryTracker* tracker, uint64_t mr) {
    uint64_t hash = hash_function(mr);
    HashNode* current = tracker->seen_buckets[hash];
    
    while (current) {
        if (current->value == mr) return true;
        current = current->next;
    }
    return false;
}

static void mark_mr_as_seen(DiscoveryTracker* tracker, uint64_t mr) {
    uint64_t hash = hash_function(mr);
    HashNode* new_node = safe_malloc(sizeof(HashNode), "seen marker");
    
    new_node->value = mr;
    new_node->next = tracker->seen_buckets[hash];
    tracker->seen_buckets[hash] = new_node;
}

static void add_unique_discovery(DiscoveryTracker* tracker, uint64_t mr, uint64_t n, uint64_t global_peak) {
    // Expand capacity if needed
    if (tracker->count >= tracker->capacity) {
        tracker->capacity *= 2;
        tracker->discoveries = safe_realloc(tracker->discoveries, 
                                          tracker->capacity * sizeof(UniqueDiscovery), 
                                          "discoveries expansion");
    }
    
    UniqueDiscovery* discovery = &tracker->discoveries[tracker->count];
    discovery->mr = mr;
    discovery->first_n = n;
    discovery->global_peak = global_peak;
    discovery->is_known = is_known_mr(mr);
    
    tracker->count++;
    
    // Show detailed message for first MAX_DETAILED_DISCOVERIES
    if (tracker->count <= MAX_DETAILED_DISCOVERIES) {
        if (discovery->is_known) {
            printf(" [*] New unique value of mr = %lu generated by n = %lu (Unique mr found: %d) - [WELL-KNOWN]\n", 
                   mr, n, tracker->count);
        } else {
            printf(" [*] New unique value of mr = %lu generated by n = %lu (Unique mr found: %d) - [NEW DISCOVERY!]\n", 
                   mr, n, tracker->count);
        }
        fflush(stdout);
    }
    
    mark_mr_as_seen(tracker, mr);
}

static bool track_unique_discovery_if_new(DiscoveryTracker* tracker, uint64_t mr, uint64_t n, uint64_t global_peak) {
    omp_set_lock(&tracker->lock);
    
    bool is_new = !has_mr_been_seen(tracker, mr);
    if (is_new) {
        add_unique_discovery(tracker, mr, n, global_peak);
    }
    
    omp_unset_lock(&tracker->lock);
    return is_new;
}

/* ====================================================================================================================
 * VALIDATION RESULTS IMPLEMENTATION
 * ==================================================================================================================== */

static ValidationResults* create_validation_results(void) {
    ValidationResults* results = safe_malloc(sizeof(ValidationResults), "validation results");
    
    const int initial_capacity = 1000;
    
    results->known_capacity = initial_capacity;
    results->new_capacity = initial_capacity;
    results->violation_capacity = initial_capacity;
    
    results->known_results = safe_malloc(initial_capacity * sizeof(MrResult), "known results");
    results->new_results = safe_malloc(initial_capacity * sizeof(MrResult), "new results");
    results->violations = safe_malloc(initial_capacity * sizeof(ViolationResult), "violations");
    
    results->known_count = 0;
    results->new_count = 0;
    results->violation_count = 0;
    
    omp_init_lock(&results->lock);
    return results;
}

static void destroy_validation_results(ValidationResults* results) {
    if (!results) return;
    
    free(results->known_results);
    free(results->new_results);
    free(results->violations);
    omp_destroy_lock(&results->lock);
    free(results);
}

static void expand_results_array(void** array, int* capacity, size_t element_size) {
    *capacity *= 2;
    *array = safe_realloc(*array, (*capacity) * element_size, "results expansion");
}

static bool is_duplicate_known_result(const ValidationResults* results, uint64_t mr) {
    for (int i = 0; i < results->known_count; i++) {
        if (results->known_results[i].mr == mr) {
            return true;
        }
    }
    return false;
}

static bool is_duplicate_new_result(const ValidationResults* results, uint64_t mr) {
    for (int i = 0; i < results->new_count; i++) {
        if (results->new_results[i].mr == mr) {
            return true;
        }
    }
    return false;
}

static void add_known_mr_result(ValidationResults* results, uint64_t mr, uint64_t n, uint64_t global_peak) {
    omp_set_lock(&results->lock);
    
    if (!is_duplicate_known_result(results, mr)) {
        if (results->known_count >= results->known_capacity) {
            expand_results_array((void**)&results->known_results, 
                                &results->known_capacity, sizeof(MrResult));
        }
        
        MrResult* result = &results->known_results[results->known_count++];
        result->mr = mr;
        result->first_n = n;
        result->global_peak = global_peak;
    }
    
    omp_unset_lock(&results->lock);
}

static void add_new_mr_result(ValidationResults* results, uint64_t mr, uint64_t n, uint64_t global_peak) {
    omp_set_lock(&results->lock);
    
    if (!is_duplicate_new_result(results, mr)) {
        if (results->new_count >= results->new_capacity) {
            expand_results_array((void**)&results->new_results, 
                                &results->new_capacity, sizeof(MrResult));
        }
        
        MrResult* result = &results->new_results[results->new_count++];
        result->mr = mr;
        result->first_n = n;
        result->global_peak = global_peak;
    }
    
    omp_unset_lock(&results->lock);
}

static void add_violation_result(ValidationResults* results, uint64_t n, uint64_t mr, 
                               uint64_t local_peak, uint64_t global_peak) {
    omp_set_lock(&results->lock);
    
    if (results->violation_count >= results->violation_capacity) {
        expand_results_array((void**)&results->violations, 
                            &results->violation_capacity, sizeof(ViolationResult));
    }
    
    ViolationResult* violation = &results->violations[results->violation_count++];
    violation->n = n;
    violation->mr = mr;
    violation->local_peak = local_peak;
    violation->global_peak = global_peak;
    
    printf(" [!] VIOLATION DETECTED: n = %lu, mr = %lu, local_peak = %lu > global_peak = %lu\n", 
           n, mr, local_peak, global_peak);
    fflush(stdout);
    
    omp_unset_lock(&results->lock);
}

/* ====================================================================================================================
 * PROGRESS TRACKER IMPLEMENTATION
 * ==================================================================================================================== */

static ProgressTracker* create_progress_tracker(void) {
    ProgressTracker* tracker = safe_malloc(sizeof(ProgressTracker), "progress tracker");
    tracker->processed = 0;
    tracker->found_count = 0;
    tracker->valid_mr_count = 0;
    tracker->last_progress_time = 0.0;
    omp_init_lock(&tracker->lock);
    return tracker;
}

static void destroy_progress_tracker(ProgressTracker* tracker) {
    if (tracker) {
        omp_destroy_lock(&tracker->lock);
        free(tracker);
    }
}

static void update_progress_counters(ProgressTracker* tracker, bool mr_found, bool is_valid) {
    #pragma omp atomic
    tracker->processed++;
    
    if (mr_found) {
        #pragma omp atomic
        tracker->found_count++;
        
        if (is_valid) {
            #pragma omp atomic
            tracker->valid_mr_count++;
        }
    }
}

static void update_progress_display(ProgressTracker* tracker, uint64_t max_n, double start_time, 
                                   const DiscoveryTracker* discovery, const ValidationResults* results) {
    omp_set_lock(&tracker->lock);
    
    double current_time = omp_get_wtime();
    
    if (current_time - tracker->last_progress_time >= PROGRESS_UPDATE_INTERVAL) {
        tracker->last_progress_time = current_time;
        
        double elapsed = current_time - start_time;
        double rate = tracker->processed / elapsed;
        double eta = (max_n - tracker->processed) / rate;
        double progress_percent = (double)tracker->processed / max_n * 100.0;
        
        printf("Progress: (%.1f%%) | Processed: %lu | Valid mr found: %lu | Unique mr found: %d | New mr found: %d | Violations: %d | %.1f nums/sec | ETA: %.1f min\n",
               progress_percent, tracker->processed, tracker->valid_mr_count, 
               discovery->count, results->new_count, results->violation_count, 
               rate, eta/60.0);
        fflush(stdout);
    }
    
    omp_unset_lock(&tracker->lock);
}

/* ====================================================================================================================
 * COLLATZ SEQUENCE ANALYSIS
 * ==================================================================================================================== */

static uint64_t find_first_mr_and_global_peak(uint64_t n_start, bool* found, uint64_t* global_peak) {
    uint64_t n = n_start;
    MValues m_values;
    init_m_values(&m_values);
    
    uint64_t first_mr = 0;
    *found = false;
    *global_peak = 0;
    
    for (int step = 0; step < MAX_SEQUENCE_LENGTH && n != 1; step++) {
        uint64_t m = calculate_m(n);
        
        if (m > *global_peak) {
            *global_peak = m;
        }
        
        if (is_m_repeated(&m_values, m)) {
            first_mr = m;
            *found = true;
            break;
        }
        
        add_m_value(&m_values, m);
        
        if (!apply_collatz_transform(&n)) {
            *found = false;
            break;
        }
        
        if (step > 10000 && n > n_start * 1000) {
            first_mr = 0;
            *found = true;
            break;
        }
    }
    
    if (n == 1 && !*found) {
        first_mr = 0;
        *found = true;
    }
    
    destroy_m_values(&m_values);
    return first_mr;
}

/* ====================================================================================================================
 * VALIDATION ENGINE IMPLEMENTATION
 * ==================================================================================================================== */

static ValidationEngine* create_validation_engine(void) {
    ValidationEngine* engine = safe_malloc(sizeof(ValidationEngine), "validation engine");
    
    engine->known_dict = KNOWN_MR_DICT;
    engine->dict_size = KNOWN_MR_COUNT;
    engine->discovery = create_discovery_tracker();
    engine->results = create_validation_results();
    engine->progress = create_progress_tracker();
    
    return engine;
}

static void destroy_validation_engine(ValidationEngine* engine) {
    if (!engine) return;
    
    destroy_discovery_tracker(engine->discovery);
    destroy_validation_results(engine->results);
    destroy_progress_tracker(engine->progress);
    free(engine);
}

static void process_mr_validation(ValidationEngine* engine, uint64_t n, uint64_t mr, uint64_t global_peak) {
    // Track unique discovery
    bool is_new_discovery = track_unique_discovery_if_new(engine->discovery, mr, n, global_peak);
    
    // Validate against known dictionary
    if (is_known_mr(mr)) {
        uint64_t local_peak = find_known_mr_local_peak(mr);
        
        if (local_peak <= global_peak) {
            add_known_mr_result(engine->results, mr, n, global_peak);
            update_progress_counters(engine->progress, true, true);
        } else {
            add_violation_result(engine->results, n, mr, local_peak, global_peak);
            update_progress_counters(engine->progress, true, false);
        }
    } else {
        add_new_mr_result(engine->results, mr, n, global_peak);
        update_progress_counters(engine->progress, true, true);
        
        // Show message for new discoveries if not already shown as unique discovery
        if (engine->discovery->count > MAX_DETAILED_DISCOVERIES) {
            printf(" [*] NEW MR DISCOVERED: mr = %lu (global_peak = %lu) generated by n = %lu\n", 
                   mr, global_peak, n);
            fflush(stdout);
        }
    }
}

static void process_single_number(uint64_t n, ValidationEngine* engine) {
    bool mr_found = false;
    uint64_t global_peak = 0;
    uint64_t mr = find_first_mr_and_global_peak(n, &mr_found, &global_peak);
    
    if (mr_found && mr >= 1) {  // Changed from mr >= 2 to mr >= 1 to include mr=1
        process_mr_validation(engine, n, mr, global_peak);
    } else {
        update_progress_counters(engine->progress, false, false);
    }
}

/* ====================================================================================================================
 * SCHEDULING CONFIGURATION
 * ==================================================================================================================== */

static SchedulingConfig configure_scheduling(uint64_t max_n) {
    SchedulingConfig config;
    config.num_threads = omp_get_max_threads();
    
    if (max_n < SMALL_RANGE_THRESHOLD) {
        config.strategy = SCHEDULE_STATIC;
        config.chunk_size = 0;
    } else if (max_n < MEDIUM_RANGE_THRESHOLD) {
        config.strategy = SCHEDULE_GUIDED;
        config.chunk_size = 0;
    } else {
        config.strategy = SCHEDULE_DYNAMIC;
        config.chunk_size = max_n / (config.num_threads * 10);
        if (config.chunk_size < MIN_CHUNK_SIZE) config.chunk_size = MIN_CHUNK_SIZE;
        if (config.chunk_size > MAX_CHUNK_SIZE) config.chunk_size = MAX_CHUNK_SIZE;
    }
    
    return config;
}

static void print_scheduling_info(const SchedulingConfig* config) {
    const char* strategy_names[] = {
        "static (small range)\n\n", 
        "guided (medium range)\n\n", 
        "dynamic (large range)\n\n"
    };
    
    printf("\nUsing scheduling strategy: %s", strategy_names[config->strategy]);
    if (config->strategy == SCHEDULE_DYNAMIC) {
        printf(" with chunk size %lu", config->chunk_size);
    }
}

/* ====================================================================================================================
 * PARALLEL EXECUTION ENGINE
 * ==================================================================================================================== */

static void execute_parallel_validation_loop(uint64_t max_n, ValidationEngine* engine, 
                                            double start_time, const SchedulingConfig* config) {
    switch (config->strategy) {
        case SCHEDULE_STATIC:
            #pragma omp parallel
            {
                int thread_num = omp_get_thread_num();
                uint64_t local_processed = 0;
                
                #pragma omp for schedule(static)
                for (uint64_t n = 1; n < max_n; n++) {
                    process_single_number(n, engine);
                    local_processed++;
                    
                    if (thread_num == 0 && local_processed % PROGRESS_CHECK_FREQUENCY == 0) {
                        update_progress_display(engine->progress, max_n, start_time, 
                                              engine->discovery, engine->results);
                    }
                }
            }
            break;
            
        case SCHEDULE_GUIDED:
            #pragma omp parallel
            {
                int thread_num = omp_get_thread_num();
                uint64_t local_processed = 0;
                
                #pragma omp for schedule(guided)
                for (uint64_t n = 1; n < max_n; n++) {
                    process_single_number(n, engine);
                    local_processed++;
                    
                    if (thread_num == 0 && local_processed % PROGRESS_CHECK_FREQUENCY == 0) {
                        update_progress_display(engine->progress, max_n, start_time, 
                                              engine->discovery, engine->results);
                    }
                }
            }
            break;
            
        case SCHEDULE_DYNAMIC:
            #pragma omp parallel
            {
                int thread_num = omp_get_thread_num();
                uint64_t local_processed = 0;
                
                #pragma omp for schedule(dynamic, config->chunk_size)
                for (uint64_t n = 1; n < max_n; n++) {
                    process_single_number(n, engine);
                    local_processed++;
                    
                    if (thread_num == 0 && local_processed % PROGRESS_CHECK_FREQUENCY == 0) {
                        update_progress_display(engine->progress, max_n, start_time, 
                                              engine->discovery, engine->results);
                    }
                }
            }
            break;
    }
}

static void execute_validation_search(uint64_t max_n, ValidationEngine* engine, double start_time) {
    SchedulingConfig config = configure_scheduling(max_n);
    print_scheduling_info(&config);
    
    execute_parallel_validation_loop(max_n, engine, start_time, &config);
}

/* ====================================================================================================================
 * RESULTS SORTING AND OUTPUT
 * ==================================================================================================================== */

static int compare_mr_results(const void* a, const void* b) {
    const MrResult* result_a = (const MrResult*)a;
    const MrResult* result_b = (const MrResult*)b;
    
    if (result_a->mr < result_b->mr) return -1;
    if (result_a->mr > result_b->mr) return 1;
    return 0;
}

static void sort_validation_results(ValidationResults* results) {
    if (results->known_count > 1) {
        qsort(results->known_results, results->known_count, sizeof(MrResult), compare_mr_results);
    }
}

static void write_csv_output(FILE* output, const ValidationResults* results) {
    fprintf(output, "n,mr,type,local_peak,global_peak,status\n");
    
    // Known mr results
    for (int i = 0; i < results->known_count; i++) {
        const MrResult* result = &results->known_results[i];
        uint64_t local_peak = find_known_mr_local_peak(result->mr);
        fprintf(output, "%lu,%lu,known,%lu,%lu,valid\n", 
                result->first_n, result->mr, local_peak, result->global_peak);
    }
    
    // New mr results
    for (int i = 0; i < results->new_count; i++) {
        const MrResult* result = &results->new_results[i];
        fprintf(output, "%lu,%lu,new,unknown,%lu,new_discovery\n", 
                result->first_n, result->mr, result->global_peak);
    }
    
    // Violations
    for (int i = 0; i < results->violation_count; i++) {
        const ViolationResult* violation = &results->violations[i];
        fprintf(output, "%lu,%lu,violation,%lu,%lu,violation\n", 
                violation->n, violation->mr, violation->local_peak, violation->global_peak);
    }
}

static void print_validation_results(const ValidationResults* results) {
    printf("\nVALIDATION RESULTS:\n\n");
    
    if (results->known_count > 0) {
        printf("Known mr values validated (%d):\n\n", results->known_count);
        for (int i = 0; i < results->known_count; i++) {
            const MrResult* result = &results->known_results[i];
            uint64_t local_peak = find_known_mr_local_peak(result->mr);
            printf(" [*] n = %lu, mr = %lu, local_peak <= global_peak (%lu <= %lu) - [VALID]\n",
                   result->first_n, result->mr, local_peak, result->global_peak);
        }
    }
    
    if (results->new_count > 0) {
        printf("NEW mr values discovered (%d) - THEORETICAL IMPOSSIBILITY:\n", results->new_count);
        for (int i = 0; i < results->new_count; i++) {
            const MrResult* result = &results->new_results[i];
            printf(" [!] mr = %lu (NOT IN DICTIONARY) generated by n = %lu - [NEW DISCOVERY]\n",
                   result->mr, result->first_n);
        }
    }
    
    if (results->violation_count > 0) {
        printf("VIOLATIONS detected (%d): local_peak > global_peak:\n", results->violation_count);
        for (int i = 0; i < results->violation_count; i++) {
            const ViolationResult* violation = &results->violations[i];
            printf(" [!] VIOLATION: n = %lu, mr = %lu, local_peak = %lu > global_peak = %lu\n",
                   violation->n, violation->mr, violation->local_peak, violation->global_peak);
        }
    }
}

static void print_final_summary(int exponent, double total_time, const ValidationEngine* engine) {
    const ProgressTracker* progress = engine->progress;
    const ValidationResults* results = engine->results;
    const DiscoveryTracker* discovery = engine->discovery;
    
    printf("\nFinal Summary:\n\n");

    printf(" [*] Exponent: %d\n", exponent);
    printf(" [*] Total time: %.2f seconds\n", total_time);
    printf(" [*] Speed: %.1f numbers/second\n", (double)progress->processed / total_time);
    printf(" [*] Numbers processed: 2^%d = %lu\n", exponent, progress->processed);
    printf(" [*] Numbers with mr > 0 found: %lu\n", progress->found_count);  // Changed from mr > 1 to mr > 0
    printf(" [*] Valid mr validations: %lu\n", progress->valid_mr_count);
    printf(" [*] Unique mr values discovered: %d\n", discovery->count);
    printf(" [*] Percentage with mr > 0: %.2f%%\n", (double)progress->found_count / progress->processed * 100.0);
    printf(" [*] Violations detected: %d\n\n", results->violation_count);

    if (results->known_count > 0) {
        printf("Final list of mr values:\n\n");
        for (int i = 0; i < results->known_count; i++) {
            if (i > 0) printf(", ");
            printf("%lu", results->known_results[i].mr);
        }
        printf("\n");
    }
    
    if (results->new_count > 0) {
        printf("New mr values discovered:\n\n");
        for (int i = 0; i < results->new_count; i++) {
            if (i > 0) printf(", ");
            printf("%lu", results->new_results[i].mr);
        }
        printf("\n");
    }
}

/* ====================================================================================================================
 * COMMAND LINE ARGUMENT PROCESSING
 * ==================================================================================================================== */

static bool validate_arguments(int argc, char* argv[], int* exponent, uint64_t* max_n) {
    if (argc != 2) {
        printf("Usage: %s <exponent>\n", argv[0]);
        printf("Example: %s 25  (to search n < 2^25)\n", argv[0]);
        printf("Recommended exponents:\n");
        printf("  20 -> 2^20 = 1,048,576 (quick test)\n");
        printf("  25 -> 2^25 = 33,554,432 (default)\n");
        printf("  30 -> 2^30 = 1,073,741,824 (intensive use)\n");
        return false;
    }
    
    *exponent = atoi(argv[1]);
    
    if (*exponent < 1 || *exponent > 64) {
        printf("Error: Exponent must be between 1 and 64\n");
        printf("Exponent %d is out of valid range\n", *exponent);
        return false;
    }
    
    *max_n = 1UL << *exponent;
    return true;
}

static void print_program_header(int exponent, uint64_t max_n) {
    printf("\n--------------------------------------------------------------------------------------------------\n");
    printf("- Bruteforce Collatz sequences using Tuple-based transform using local and global peaks approach -\n");
    printf("--------------------------------------------------------------------------------------------------\n");

    printf("\nUsing %d threads for parallelization\n", omp_get_max_threads());
    printf("Validating mr values using known dictionary between sequences of n < 2^%d...\n", exponent);
    printf("Range: from 1 to %lu\n", max_n - 1);
    printf("Validation rule: local_peak <= global_peak for each mr found\n");
}

static FILE* create_output_file(int exponent) {
    char filename[256];
    snprintf(filename, sizeof(filename), "mr_validation_results_2_%d.txt", exponent);
    
    FILE* output = fopen(filename, "w");
    if (!output) {
        printf("Error: Could not create the output file %s\n", filename);
    }
    return output;
}

/* ====================================================================================================================
 * MAIN FUNCTION
 * ==================================================================================================================== */

int main(int argc, char* argv[]) {
    // Parse and validate command line arguments
    int exponent;
    uint64_t max_n;
    if (!validate_arguments(argc, argv, &exponent, &max_n)) {
        return 1;
    }
    
    // Print program header
    print_program_header(exponent, max_n);
      
    // Initialize validation engine
    ValidationEngine* engine = create_validation_engine();
    
    // Create output file
    FILE* output = create_output_file(exponent);
    if (!output) {
        destroy_validation_engine(engine);
        return 1;
    }
    
    // Execute validation search
    double start_time = omp_get_wtime();
    execute_validation_search(max_n, engine, start_time);
    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;
    
    // Sort and write results
    sort_validation_results(engine->results);
    write_csv_output(output, engine->results);
    fflush(output);
    
    // Display results
    print_validation_results(engine->results);
    print_final_summary(exponent, total_time, engine);
    
    // Clean up
    fclose(output);
    printf("\nResults saved to 'mr_validation_results_2_%d.txt'\n", exponent);
    
    destroy_validation_engine(engine);
    
    return 0;
}