// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.0
//
// MAC4 v2 datapath: the original 4-lane int8 dot product (MAC4), plus an
// internal "stationary" input buffer and T=8 parallel 32-bit accumulators
// used by the 4 new instructions (LOAD_STATIONARY, MAC_TILED, READ_ACC,
// RESET_ACC -- see mac4_instr_pkg.sv for their semantics/encoding).
//
// All 5 instructions share the same 1-cycle pipelined skeleton: next-state
// computed combinationally, registered through a single pipeline stage
// (1-cycle latency, fully pipelined at 1 op/cycle). The stationary buffer
// and accumulators are the only state that persists *between* instructions;
// everything else (result/hartid/id/rd/valid/we) is per-instruction, exactly
// as in the original MAC4-only mac4_alu.sv.

module mac4_alu
  import mac4_instr_pkg::*;
#(
    parameter int unsigned NrRgprPorts = 2,
    parameter int unsigned XLEN = 32,
    parameter type hartid_t = logic,
    parameter type id_t = logic,
    parameter type registers_t = logic
) (
    input  logic                  clk_i,
    input  logic                  rst_ni,
    input  registers_t            registers_i,
    input  opcode_t               opcode_i,
    input  logic       [     6:0] funct7_i,
    input  hartid_t                hartid_i,
    input  id_t                   id_i,
    input  logic       [     4:0] rd_i,
    output logic       [XLEN-1:0] result_o,
    output hartid_t               hartid_o,
    output id_t                   id_o,
    output logic       [     4:0] rd_o,
    output logic                  valid_o,
    output logic                  we_o
);

  // T = number of parallel output-channel accumulators (tiling factor).
  // StationaryWords = depth of the stationary input buffer, sized to the
  // 6-bit word index carried in funct7_i[5:0]/funct7_i (0..63), comfortably
  // covering the largest segment encountered (~38 words for fc2).
  localparam int unsigned T = 8;
  localparam int unsigned StationaryWords = 64;

  logic [XLEN-1:0] result_n, result_q;
  hartid_t hartid_n, hartid_q;
  id_t id_n, id_q;
  logic valid_n, valid_q;
  logic [4:0] rd_n, rd_q;
  logic we_n, we_q;

  // Persistent coprocessor state: survives across instructions.
  logic [31:0] stationary_n[StationaryWords], stationary_q[StationaryWords];
  logic signed [31:0] acc_n[T], acc_q[T];

  assign result_o = result_q;
  assign hartid_o = hartid_q;
  assign id_o     = id_q;
  assign valid_o  = valid_q;
  assign rd_o     = rd_q;
  assign we_o     = we_q;

  // Word index into the stationary buffer (LOAD_STATIONARY writes it,
  // MAC_TILED reads it) and accumulator slot indices. rd_i[2:0] is reused as
  // MAC_TILED's target slot: MAC_TILED has no real destination register
  // (writeback = 0), so the rd field only ever carries this slot index.
  logic [5:0] mot_idx;
  logic [2:0] mac_tiled_slot;
  logic [2:0] read_acc_slot;
  assign mot_idx        = funct7_i[5:0];
  assign mac_tiled_slot = rd_i[2:0];
  assign read_acc_slot  = funct7_i[2:0];

  // Shared 4-lane int8 dot product, used by both MAC4 (rs1 x rs2) and
  // MAC_TILED (stationary[mot] x rs1).
  logic [31:0] dot_input_word, dot_weight_word;
  always_comb begin
    unique case (opcode_i)
      MAC4:      begin
        dot_input_word  = registers_i[0];
        dot_weight_word = registers_i[1];
      end
      MAC_TILED: begin
        dot_input_word  = stationary_q[mot_idx];
        dot_weight_word = registers_i[0];
      end
      default:   begin
        dot_input_word  = '0;
        dot_weight_word = '0;
      end
    endcase
  end

  logic signed [8:0] input_lane[4];  // zero-extended uint8: always in 0..255
  logic signed [7:0] weight_lane[4];  // sign-extended int8: -128..127
  logic signed [17:0] product[4];  // max magnitude 255*128 = 32640, fits in 17 bits

  for (genvar i = 0; i < 4; i++) begin : gen_lanes
    assign input_lane[i]  = $signed({1'b0, dot_input_word[8*i+:8]});
    assign weight_lane[i] = $signed(dot_weight_word[8*i+:8]);
    assign product[i]     = input_lane[i] * weight_lane[i];
  end

  logic signed [31:0] dot_sum;
  assign dot_sum = $signed(product[0]) + $signed(product[1]) + $signed(product[2]) + $signed(product[3]);

  always_comb begin
    result_n     = '0;
    hartid_n     = '0;
    id_n         = '0;
    valid_n      = '0;
    rd_n         = '0;
    we_n         = '0;
    stationary_n = stationary_q;
    acc_n        = acc_q;

    case (opcode_i)
      MAC4: begin
        result_n = dot_sum;
        hartid_n = hartid_i;
        id_n     = id_i;
        valid_n  = 1'b1;
        rd_n     = rd_i;
        we_n     = 1'b1;
      end
      LOAD_STATIONARY: begin
        stationary_n[mot_idx] = registers_i[0];
        hartid_n = hartid_i;
        id_n     = id_i;
        valid_n  = 1'b1;
        rd_n     = rd_i;
        we_n     = 1'b0;
      end
      MAC_TILED: begin
        acc_n[mac_tiled_slot] = acc_q[mac_tiled_slot] + dot_sum;
        hartid_n = hartid_i;
        id_n     = id_i;
        valid_n  = 1'b1;
        rd_n     = rd_i;
        we_n     = 1'b0;
      end
      READ_ACC: begin
        result_n = acc_q[read_acc_slot];
        hartid_n = hartid_i;
        id_n     = id_i;
        valid_n  = 1'b1;
        rd_n     = rd_i;
        we_n     = 1'b1;
      end
      RESET_ACC: begin
        for (int unsigned i = 0; i < T; i++) acc_n[i] = '0;
        hartid_n = hartid_i;
        id_n     = id_i;
        valid_n  = 1'b1;
        rd_n     = rd_i;
        we_n     = 1'b0;
      end
      default: begin
        result_n = '0;
        hartid_n = '0;
        id_n     = '0;
        valid_n  = '0;
        rd_n     = '0;
        we_n     = '0;
      end
    endcase
  end

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      result_q     <= '0;
      hartid_q     <= '0;
      id_q         <= '0;
      valid_q      <= '0;
      rd_q         <= '0;
      we_q         <= '0;
      stationary_q <= '{default: '0};
      acc_q        <= '{default: '0};
    end else begin
      result_q     <= result_n;
      hartid_q     <= hartid_n;
      id_q         <= id_n;
      valid_q      <= valid_n;
      rd_q         <= rd_n;
      we_q         <= we_n;
      stationary_q <= stationary_n;
      acc_q        <= acc_n;
    end
  end

endmodule
