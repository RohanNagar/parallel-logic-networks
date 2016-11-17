`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/10/2016 02:34:59 PM
// Design Name: 
// Module Name: Combinational
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module Combinational(iA,iB, oD);
    input iA,iB;
    output oD;
    
    assign oD = iA + iB;  
endmodule
