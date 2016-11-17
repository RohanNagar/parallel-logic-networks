module fourBitSubtractor(D, Bout, A, B, Bin);
output [4:1] D;
output Bout;
input [3:0] A, B;
input Bin;

wire [3:1] C;

oneBitSubtractor S0 (A[0], B[0], Bin, C[1], D[1]);
oneBitSubtractor S1 (A[1], B[1], C[1], C[2], D[2]);
oneBitSubtractor S2 (A[2], B[2], C[2], C[3], D[3]);
oneBitSubtractor S3 (A[3], B[3], C[3], Bout, D[4]);

endmodule
