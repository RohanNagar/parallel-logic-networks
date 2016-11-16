

// graph to matrix function 
// takes g as input from graph
void graphToMatrix(graph &g){

  gateMatrix matrix = gateMatrix(g.get_max_height(), g.get_max_width(), top.get_input().size(),
                                 top.get_output().size() );

  for(int i = 0; i < g.get_gate_list().size(); i++){
   // i contains the index of the current gate value

   // grab first gate id input 0 from graph
   gid_t input0 = gate.get_graph()[i][0];
   gid_t input1;
  
   // if gate has second input
   if(gate.get_graph().size() > 1){
     // grab seccond gate id input 1 from graph
     input1 =  gate.get_graph()[i][1];
   }
   else{
     input1 = 0;
   }

   matrix.addGate(g.get_gate_list()[i].get_height(), g.get_gate_list()[i].get_width(),
                  g.get_gate_list()[i].get_gate_type(), g.get_gate_list()[input0].get_height(), 
                  g.get_gate_list()[input0].get_width(), g.get_gate_list()[input1].get_height(),
                  g.get_gate_list()[input1].get_width());
  }

  matrix.printMatrix();
}
