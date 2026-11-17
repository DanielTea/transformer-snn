/* Coasting read heads on a tape of the byte stream.

   A head is one neuron sitting on a tape cell. Every byte it moves one cell on
   and predicts the byte stored there. A confirmed byte adds one unit of charge,
   a wrong byte drains or kills the head. Key neurons hold one synapse each onto
   a tape cell. A byte that arrives with error spikes moves the synapse of the
   previous context's key neuron onto its own cell. A lookup places a head on
   the cell that the current context's key neuron points at.

   Everything is integer and causal: the prediction for byte t + 1 uses bytes
   0..t only.
*/
#include <stdint.h>
#include <stdlib.h>

static inline uint32_t context_key(const uint8_t *bytes, int order, uint32_t seed) {
    uint32_t h = seed;
    for (int i = 0; i < order; i++) h = (h ^ bytes[i]) * 16777619u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return h;
}

/* data[0..n-1], surprise[t] = error spikes left when byte t arrived (t < n - 1).
   pred[t]   byte the head predicts for data[t + 1], or -1 when no head is alive
   charge[t] confirmed bytes the head carries at that moment
   order        context bytes of the key neurons
   cells        key neurons, a power of two
   gate         error spikes needed to move a synapse, 0 moves one on every byte
   tape         cells of the ring, 0 for an unbounded tape
   lookup_gate  error spikes needed to place a head, 0 places one whenever none is alive
   miss         0: a wrong byte kills the head. k > 0: charge >>= k and the head moves on while charge stays positive
   relock_below an error spike replaces a live head that carries less charge than this, 0 for never
   lifetime     bytes a head predicts before it is removed, 0 for as long as it is right
   stats[0..3]  synapse moves, lookups, placements, rows with a live head */
void coast(const uint8_t *data, const int16_t *surprise, int64_t n, int order, int64_t cells, int gate,
           int64_t tape, int lookup_gate, int miss, int relock_below, uint32_t seed, int lifetime,
           int16_t *pred, int32_t *charge, int64_t *stats) {
    int64_t *synapse = malloc(sizeof(int64_t) * cells);
    for (int64_t i = 0; i < cells; i++) synapse[i] = -1;
    int64_t pointer = -1, moves = 0, lookups = 0, placements = 0, alive_rows = 0;
    int32_t held = 0, emitted = 0;
    for (int64_t t = 0; t < n - 1; t++) {
        if (pointer >= 0) {
            if (lifetime > 0 && emitted >= lifetime) { pointer = -1; held = 0; }
            else if (data[pointer] == data[t]) { if (held < 1000000) held++; pointer++; }
            else if (miss > 0 && (held >> miss) > 0) { held >>= miss; pointer++; }
            else { pointer = -1; held = 0; }
        }
        if (t >= order && surprise[t] >= gate) {
            synapse[context_key(data + t - order, order, seed) & (cells - 1)] = t;
            moves++;
        }
        int wants = pointer < 0 || (relock_below > 0 && held < relock_below && surprise[t] >= 1);
        if (wants && t + 1 >= order && surprise[t] >= lookup_gate) {
            lookups++;
            int64_t cell = synapse[context_key(data + t + 1 - order, order, seed) & (cells - 1)];
            if (cell >= 0) {
                /* The ring keeps the last `tape` bytes. An older synapse reaches whatever now sits in its cell. */
                if (tape > 0) cell = t - ((t - cell) % tape);
                pointer = cell; held = 0; emitted = 0; placements++;
            }
        }
        if (pointer >= 0) { pred[t] = data[pointer]; charge[t] = held; emitted++; alive_rows++; }
        else { pred[t] = -1; charge[t] = 0; }
    }
    stats[0] = moves; stats[1] = lookups; stats[2] = placements; stats[3] = alive_rows;
    free(synapse);
}
