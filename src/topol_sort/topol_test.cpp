#include <iostream>
#include <list>
#include <stack>
#include "gate.h"
#include "graph.h"
#include "module.h"
using namespace pln;
using namespace std;
 
// Class to represent a graph
/*class Graph
{
    int V;    // No. of vertices'
 
    // Pointer to an array containing adjacency listsList
    list<int> *adj;
 
    // A function used by topologicalSort
    void topologicalSortUtil(int v, bool visited[], stack<int> &Stack);
public:
    Graph(int V);   // Constructor
 
     // function to add an edge to graph
    void addEdge(int v, int w);
 
    // prints a Topological Sort of the complete graph
    void topologicalSort();
};
 
Graph::Graph(int V)
{
    this->V = V;
    adj = new list<int>[V];
}
 
void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w); // Add w to vs list.
}
 */

// A recursive function used by topologicalSort
void topologicalSortUtil(graph& g, int v, bool visited[], 
                                stack<int> &Stack)
{
    // Mark the current node as visited.
    visited[v] = true;

    // Recur for all the vertices adjacent to this vertex
    if(g.get_graph()[v].size() > 0){
      for (int i = 0; i < (g.get_graph()[v]).size(); ++i){
          if (!visited[g.get_graph()[v][i]]){
              topologicalSortUtil(g, g.get_graph()[v][i], visited, Stack);
          }
      }
    }
    // Push current vertex to stack which stores result
    Stack.push(v);
}
 

// The function to do Topological Sort. It uses recursive 
// topologicalSortUtil()
void topologicalSort(graph& g)
{
    stack<int> Stack;

    // Mark all the vertices as not visited
    bool *visited = new bool[g.get_gate_list().size()];
    for (int i = 0; i < g.get_gate_list().size(); i++){
        visited[i] = false;
    }

    // Call the recursive helper function to store Topological
    // Sort starting from all vertices one by one
    for (int i = 0; i < g.get_gate_list().size(); i++){
      if (visited[i] == false){
        topologicalSortUtil(g, i, visited, Stack);
      }
    }


    // CREATE FUNCTION FOR THIS: sets heights 
    // Print contents and set heights of stack
    uint32_t Max_Level = 0; // add this value to graph
    uint32_t Max_Width = 0; // add this value to graph
    vector<uint32_t> width; width.push_back(0);    

    // Creating levels and width
    vector<gate>& cur_gate_list = g.get_gate_list();
    vector<vector<gtid_t>>& cur_graph =  g.get_graph();   
    
    // Visited 
    // Mark all the vertices as not visited
    delete visited;
    visited = new bool[g.get_gate_list().size()];
    for (int i = 0; i < g.get_gate_list().size(); i++){
        visited[i] = false;
    }

    while (Stack.empty() == false)
    {
        cout << Stack.top() << " ";

        gtid_t id = Stack.top();
        gate& cur_gate = cur_gate_list[id];

        // set its own height if it is PORT_O root
        if(cur_gate.get_type() == "PORT_O"){        
          cur_gate.set_gate_level(0); 
          cur_gate.set_gate_pos(width[0]);
          width[0]++;      
        }
   
        // set its children's height based on its own with comparison
        if(cur_graph[id].size() > 0){
          
          // CREATE FUNCTION FOR THIS

          // set first child
          gate& input0 = cur_gate_list[cur_graph[id][0]];
           
          // if input level is smaller than a parent gate + 1 (initialize all heights to 0)
          if(input0.get_gate_level() < cur_gate.get_gate_level() + 1){
cout << "input0 id: " << input0.get_id();

            // create new level on width if the desired level is not on the list
            if(cur_gate.get_gate_level() + 1 == width.size()){
cout << " new level, ";
              width.push_back(0);
            }
  
            // remove from previous position allocation;          
            if(visited[input0.get_id()]){
cout << "visited previously,";
              width[input0.get_gate_level()]--;
            }
        
            // give it its new height and position
            input0.set_gate_level(cur_gate.get_gate_level() + 1);
            input0.set_gate_pos(width[cur_gate.get_gate_level() + 1]);
            width[input0.get_gate_level()]++;
cout << " level " << input0.get_gate_level() << " pos " << input0.get_gate_pos() << "\n";
            visited[input0.get_id()] = true;            
          }
          
          // set second child
          if(cur_graph[id].size() == 2){
            gate& input1 = cur_gate_list[cur_graph[id][1]];
cout << "  input1 id: " << input1.get_id();

            // if input level is smaller than a parent gate + 1 (initialize all heights to 0)
            if(input1.get_gate_level() < cur_gate.get_gate_level() + 1){
            
              // create new level on width if the desired level is not on the list
              if(cur_gate.get_gate_level() + 1 == width.size()){
cout << " new level, "; 
                width.push_back(0);
              }

              // remove from previous position allocation;
              if(visited[input1.get_id()]){
                width[input1.get_gate_level()]--;
cout << "visited previously,";
              }

              // give it its new height and position
              input1.set_gate_level(cur_gate.get_gate_level() + 1);
              input1.set_gate_pos(width[cur_gate.get_gate_level() + 1]);
              width[input1.get_gate_level()]++;
cout << " level " << input1.get_gate_level() << " pos " << input1.get_gate_pos() << "\n";
              visited[input1.get_id()] = true;
            }
          }
       }   
       Stack.pop();
    }
    // AFTER THIS, GO TO TOP MODULES INPUT AND FORCE IT TO MAX LEVEL
}
 
// Driver program to test above functions
int main()
{
    // Create a graph given in the above diagram 
    /*
    Graph g(6);
    g.addEdge(5, 2);
    g.addEdge(5, 0);
    g.addEdge(4, 0);
    g.addEdge(4, 1);
    g.addEdge(2, 3);
    g.addEdge(3, 1);
    */

    // create graph
    graph g{};
    
    // create gates
    gate PO_30  = gate{"PO_30",  "PORT_O"}; g.insert_gate(PO_30);
    gate PI_00  = gate{"PI_00",  "PORT_I"}; g.insert_gate(PI_00);
    gate PI_01  = gate{"PI_01",  "PORT_I"}; g.insert_gate(PI_01);
    gate AND_10 = gate{"AND_10", "AND"};    g.insert_gate(AND_10);
    gate BUF_20 = gate{"BUF_20", "OBUF"};   g.insert_gate(BUF_20);

    g.insert_edge(3,1);
    g.insert_edge(3,2);
    g.insert_edge(4,3);
    g.insert_edge(0,4);

    g.print();

   
    cout << "Following is a Topological Sort of the given graph \n";
    topologicalSort(g);
    g.print(); 
    return 0;
}
