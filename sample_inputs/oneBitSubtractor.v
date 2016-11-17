module oneBitSubtractor(A, B, Bin, Bout, Diff);
output Diff, Bout;
input A, B, Bin;

assign #10 Diff = A ^ B ^ Bin;
assign #10 Bout = (B && Bin) || (~A && Bin) || (~A && B); 

endmodule
