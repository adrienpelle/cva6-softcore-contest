// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.0
//
// MAC4 v2 custom instruction table, consumed by the generic
// core/cvxif_example/instr_decoder.sv (table-driven, reused as-is).
// Replaces cvxif_example's cvxif_instr_pkg with this table.

package mac4_instr_pkg;

  typedef enum logic [2:0] {
    ILLEGAL         = 3'b000,
    MAC4            = 3'b001,
    LOAD_STATIONARY = 3'b010,
    MAC_TILED       = 3'b011,
    READ_ACC        = 3'b100,
    RESET_ACC       = 3'b101
  } opcode_t;

  typedef struct packed {
    logic accept;
    logic writeback;
    logic [2:0] register_read;
  } issue_resp_t;

  typedef struct packed {
    logic [31:0] instr;
    logic [31:0] mask;
    issue_resp_t resp;
    opcode_t     opcode;
  } copro_issue_resp_t;

  // All 5 instructions are standard R-type under opcode = custom-0 (0001011),
  // distinguished solely by funct3 (funct7 carries no opcode-selection bits
  // for the 4 new instructions: it is reused as data, read directly out of
  // issue_req.instr by mac4_coprocessor.sv since the generic instr_decoder.sv
  // does not expose it). rd (instr[11:7]) is always passed through as rd_o
  // regardless of writeback, so instructions with writeback=0 below reuse its
  // low bits as an extra data field (see mac4_alu.sv).
  //
  // MAC4 rd, rs1, rs2  (funct3 = 000, funct7 = 0000000, unchanged from v1)
  //   rd = sum_{i=0..3}( uint8(rs1[8*i +: 8]) * int8(rs2[8*i +: 8]) )
  // No rs3/accumulate operand: this CVA6's CV-X-IF issue interface only ever
  // delivers 2 register operands (ariane_pkg::NR_RGPR_PORTS = 2); MAC4's own
  // accumulation into a running sum is done in software with a plain `add`.
  //
  // LOAD_STATIONARY rs1  (funct3 = 001, funct7 = don't care)
  //   stationary[funct7[6:0]] = rs1  -- writes one already-loaded input word
  //   into the coprocessor's internal stationary buffer at word index
  //   funct7[6:0]. Buffer is built to exactly 80 words (0..79 valid) -- 4
  //   spatial positions x conv2's 20-word segment, the largest batch this
  //   effort's weight-spatial-batching (B=4) needs; funct7[6:0] can encode
  //   0..127 but software must never issue 80..127 (see mac4_alu.sv).
  //
  // MAC_TILED rs1  (funct3 = 010, funct7 = don't care)
  //   acc[rd[4:0]] += sum_{i=0..3}( uint8(stationary[funct7[6:0]][8*i +: 8])
  //                                 * int8(rs1[8*i +: 8]) )
  //   rs1 = weight word (reloaded every call, never stationary). funct7[6:0]
  //   selects the stationary word; rd[4:0] (rd has no real destination here,
  //   writeback = 0) selects the target accumulator slot (0..31, T = 32 =
  //   channel-tile(8) x spatial-batch(4)).
  //
  // READ_ACC rd  (funct3 = 011, funct7 = don't care)
  //   rd = acc[funct7[4:0]]  -- returns the current value of accumulator slot
  //   funct7[4:0] (0..31). rd here IS the real destination register.
  //
  // RESET_ACC  (funct3 = 100, funct7 = don't care)
  //   acc[i] = 0 for all i in 0..31  -- global reset of all T=32 accumulator
  //   slots, no per-slot reset (T is small enough that this is negligible).
  parameter int unsigned NbInstr = 5;
  parameter copro_issue_resp_t CoproInstr[NbInstr] = '{
      '{
          instr: 32'b0000000_00000_00000_000_00000_0001011,
          mask:  32'b1111111_00000_00000_111_00000_1111111,
          resp : '{accept: 1'b1, writeback: 1'b1, register_read: 3'b011},
          opcode: MAC4
      },
      '{
          instr: 32'b0000000_00000_00000_001_00000_0001011,
          mask:  32'b0000000_00000_00000_111_00000_1111111,
          resp : '{accept: 1'b1, writeback: 1'b0, register_read: 3'b001},
          opcode: LOAD_STATIONARY
      },
      '{
          instr: 32'b0000000_00000_00000_010_00000_0001011,
          mask:  32'b0000000_00000_00000_111_00000_1111111,
          resp : '{accept: 1'b1, writeback: 1'b0, register_read: 3'b001},
          opcode: MAC_TILED
      },
      '{
          instr: 32'b0000000_00000_00000_011_00000_0001011,
          mask:  32'b0000000_00000_00000_111_00000_1111111,
          resp : '{accept: 1'b1, writeback: 1'b1, register_read: 3'b000},
          opcode: READ_ACC
      },
      '{
          instr: 32'b0000000_00000_00000_100_00000_0001011,
          mask:  32'b0000000_00000_00000_111_00000_1111111,
          resp : '{accept: 1'b1, writeback: 1'b0, register_read: 3'b000},
          opcode: RESET_ACC
      }
  };

endpackage
