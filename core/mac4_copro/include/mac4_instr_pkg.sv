// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.0
//
// MAC4 custom instruction table, consumed by the generic
// core/cvxif_example/instr_decoder.sv (table-driven, reused as-is).
// Replaces cvxif_example's cvxif_instr_pkg with a single-entry table.

package mac4_instr_pkg;

  typedef enum logic {
    ILLEGAL = 1'b0,
    MAC4    = 1'b1
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

  // MAC4 rd, rs1, rs2
  //   rd = sum_{i=0..3}( uint8(rs1[8*i +: 8]) * int8(rs2[8*i +: 8]) )
  // Standard R-type: opcode = custom-0 (0001011), funct3 = 000, funct7 = 0000000.
  // No rs3/accumulate operand: this CVA6's CV-X-IF issue interface only ever
  // delivers 2 register operands (ariane_pkg::NR_RGPR_PORTS = 2); accumulation
  // into the running sum is done in software with a plain `add`.
  parameter int unsigned NbInstr = 1;
  parameter copro_issue_resp_t CoproInstr[NbInstr] = '{
      '{
          instr: 32'b0000000_00000_00000_000_00000_0001011,
          mask:  32'b1111111_00000_00000_111_00000_1111111,
          resp : '{accept: 1'b1, writeback: 1'b1, register_read: 3'b011},
          opcode: MAC4
      }
  };

endpackage
