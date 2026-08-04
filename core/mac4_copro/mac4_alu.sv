// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.0
//
// MAC4 arithmetic: a 4-lane int8 dot product, computed combinationally and
// registered through a single pipeline stage (1-cycle latency, fully
// pipelined at 1 op/cycle), matching the skeleton of
// core/cvxif_example/copro_alu.sv. No persistent state between instructions.

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

  logic [XLEN-1:0] result_n, result_q;
  hartid_t hartid_n, hartid_q;
  id_t id_n, id_q;
  logic valid_n, valid_q;
  logic [4:0] rd_n, rd_q;
  logic we_n, we_q;

  assign result_o = result_q;
  assign hartid_o = hartid_q;
  assign id_o     = id_q;
  assign valid_o  = valid_q;
  assign rd_o     = rd_q;
  assign we_o     = we_q;

  // registers_i[0] = rs1 (packed inputs, 4x uint8)
  // registers_i[1] = rs2 (packed weights, 4x int8)
  logic signed [8:0] input_lane[4];  // zero-extended uint8: always in 0..255
  logic signed [7:0] weight_lane[4];  // sign-extended int8: -128..127
  logic signed [17:0] product[4];  // max magnitude 255*128 = 32640, fits in 17 bits

  for (genvar i = 0; i < 4; i++) begin : gen_lanes
    assign input_lane[i]  = $signed({1'b0, registers_i[0][8*i+:8]});
    assign weight_lane[i] = $signed(registers_i[1][8*i+:8]);
    assign product[i]     = input_lane[i] * weight_lane[i];
  end

  always_comb begin
    case (opcode_i)
      MAC4: begin
        result_n = $signed(product[0]) + $signed(product[1]) + $signed(product[2]) + $signed(product[3]);
        hartid_n = hartid_i;
        id_n     = id_i;
        valid_n  = 1'b1;
        rd_n     = rd_i;
        we_n     = 1'b1;
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
      result_q <= '0;
      hartid_q <= '0;
      id_q     <= '0;
      valid_q  <= '0;
      rd_q     <= '0;
      we_q     <= '0;
    end else begin
      result_q <= result_n;
      hartid_q <= hartid_n;
      id_q     <= id_n;
      valid_q  <= valid_n;
      rd_q     <= rd_n;
      we_q     <= we_n;
    end
  end

endmodule
