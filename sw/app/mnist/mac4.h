#ifndef N2D2_EXPORTC_MAC4_H
#define N2D2_EXPORTC_MAC4_H

#include <stdint.h>

// MAC4 custom instruction: rd = sum_{i=0..3}( uint8(rs1[8*i+:8]) * int8(rs2[8*i+:8]) )
// R-type encoding: opcode = custom-0 (0x0B), funct3 = 0, funct7 = 0.
static inline int32_t mac4(uint32_t packed_inputs, uint32_t packed_weights) {
    int32_t result;
    __asm__ volatile (
        ".insn r 0x0B, 0, 0, %0, %1, %2"
        : "=r"(result)
        : "r"(packed_inputs), "r"(packed_weights)
    );
    return result;
}

#endif // N2D2_EXPORTC_MAC4_H
