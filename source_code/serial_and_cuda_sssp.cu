#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <limits.h>
#include <omp.h>
#include <nvgraph.h>

#define NODES 40000       //number of nodes
#define MIN_DEGREE 512    //minimum number of edges per node
#define INFI INT_MAX      //infinity value for distance
#define THREAD_NUM 4      //number of threads for OpenMP

void graph_initialization(int * graph, int * edge_degree_array);
void print_graph_info(int * graph);
void serial_sssp(int * graph, int src_node, long * node_distance_array, int * parent_nodes_array, int * visited_nodes_array);
void cuda_sssp(int * graph, int src_node, long * node_distance_array);


int main(int argc, char const *argv[]) {

  /* Setup necessary arrays and matrices */
  int * graph = (int *) malloc (NODES * NODES * sizeof(int));
  long * node_distance_array = (long *) malloc (NODES * sizeof(long));
  int * parent_nodes_array = (int *) malloc (NODES * sizeof(int));
  int * edge_degree_array = (int *) malloc (NODES * sizeof(int));
  int * visited_nodes_array = (int *) malloc (NODES * sizeof(int));

  /* One row for serial_sssp, One row for cuda_sssp */
  long * distance_results_matrix = (long *) malloc (2 * NODES * sizeof(long));

  /* Initialize graph */
  graph_initialization(graph, edge_degree_array);
  print_graph_info(graph);
  int src_node = (rand() % NODES);
  printf("src_node is %d\n", src_node);

  /* Start to run serial sssp */
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);
  serial_sssp(graph, src_node, node_distance_array, parent_nodes_array, visited_nodes_array);
  gettimeofday(&t2, NULL);
  double used_time = t2.tv_sec-t1.tv_sec +(t2.tv_usec-t1.tv_usec)/1000000.0;
  printf("Serial sssp used time: %f sec\n", used_time);
  /* Copy results to the result matrix for later result comparison */
  for (size_t i = 0; i < NODES; i++) {
    distance_results_matrix[0 * NODES + i] = node_distance_array[i];
  }

  /* Start to run CUDA parallel sssp */
  gettimeofday(&t1, NULL);
  cuda_sssp(graph, src_node, node_distance_array);
  gettimeofday(&t2, NULL);
  used_time = t2.tv_sec-t1.tv_sec +(t2.tv_usec-t1.tv_usec)/1000000.0;
  printf("CUDA sssp (with graph convert & transfer) used time: %f sec\n", used_time);
  /* Copy results to the result matrices for later result comparison */
  for (size_t i = 0; i < NODES; i++) {
    distance_results_matrix[1 * NODES + i] = node_distance_array[i];
  }

  /* Compare the results */
  int dis_errors = 0;
  for (size_t i = 0; i < NODES; i++) {
    if (distance_results_matrix[i] != distance_results_matrix[NODES + i])
      dis_errors += 1;
  }
  printf("dis_errors = %d\n", dis_errors);

  /* free all resources */
  free(graph);
  free(node_distance_array);
  free(parent_nodes_array);
  free(edge_degree_array);
  free(visited_nodes_array);
  free(distance_results_matrix);
  return 0;
}

/*************************** Functions for creating graph ***************************/
/* Create a connected graph that the degree density mets the MIN_DEGREE */
void graph_initialization(int * graph, int * edge_degree_array) {

  for (size_t i = 0; i < NODES; i++) {
    edge_degree_array[i] = 0;
  }

  /* 0 means no edge */
  for (size_t i = 0; i < NODES; i++) {
    for (size_t j = 0; j < NODES; j++) {
      graph[i * NODES + j] = 0;
    }
  }

  /* 1st round to add edges to form a connected graph */
  for (size_t i = 1; i < NODES; i++) {
    int random_index = (rand() % i);
    int edge_weight = (rand() % 100000) + 1; //ensure no edge weight is 0
    graph[random_index * NODES + i] = edge_weight;
    graph[i * NODES + random_index] = edge_weight;
    edge_degree_array[i] += 1;
    edge_degree_array[random_index] += 1;
  }

  /* 2nd round to add edges until the degree of each node mets the MIN_DEGREE */
  for (size_t i = 0; i < NODES; i++) {
    while (edge_degree_array[i] < MIN_DEGREE) {
      int random_index = (rand() % NODES);
      int edge_weight = (rand() % 100000) + 1; //ensure no edge weight is 0

      /* Add an edge when no edge exist and not connect to itself */
      if (graph[random_index * NODES + i] == 0 && graph[i * NODES + random_index] == 0 && random_index != i) {
        graph[random_index * NODES + i] = edge_weight;
        graph[i * NODES + random_index] = edge_weight;
        edge_degree_array[i] += 1;
        edge_degree_array[random_index] += 1;
      }
    }
  }
}

/* Print the graph matrix */
void print_graph_info(int * graph) {
  printf("Num of Nodes: %d\n", NODES);
  int num_edges = 0;
  for (size_t i = 0; i < NODES; i++) {
    for (size_t j = 0; j < NODES; j++) {
      if (j > i && graph[i * NODES + j] != 0) {
        num_edges++;
      }
    }
  }
  printf("Num of Edges: %d\n", num_edges);
}
/*************************** Functions for creating graph ***************************/


/*************************** Serial implementation for sssp ***************************/
/* extract_min for serial_sssp */
int extract_min(long * node_distance_array, int * visited_nodes_array) {
  int node;
  int dist = INFI;
  for (size_t i = 0; i < NODES; i++) {
    if (node_distance_array[i] < dist && visited_nodes_array[i] == 0) {
      node = i;
      dist = node_distance_array[i];
    }
  }
  return node;
}

/* Serial sssp, not take advantage of optimized data structure , served as baseline */
void serial_sssp(int * graph, int src_node, long * node_distance_array, int * parent_nodes_array, int * visited_nodes_array) {
  /* Initialization */
  for (size_t i = 0; i < NODES; i++) {
    node_distance_array[i] = INFI;
    parent_nodes_array[i] = -1;
    visited_nodes_array[i] = 0;
  }
  node_distance_array[src_node] = 0;

  /* Dijkstra Core */
  for (size_t i = 0; i < NODES; i++) {
    int cur_node = extract_min(node_distance_array, visited_nodes_array);
    visited_nodes_array[cur_node] = 1;
    for (int next = 0; next < NODES; next++) {
      if (graph[cur_node * NODES + next] != 0 && visited_nodes_array[next] != 1) {
        long new_distance = node_distance_array[cur_node] + graph[cur_node * NODES + next];
        /* Relaxation */
        if (new_distance < node_distance_array[next]) {
          node_distance_array[next] = new_distance;
          parent_nodes_array[next] = cur_node;
        }
      }
    }
  }
}
/*************************** Serial implementation for sssp ***************************/


/*************************** nvGraph CUDA parallel implementation for sssp ***************************/
/* nvGraph check function */
void check(nvgraphStatus_t status) {
    if (status != NVGRAPH_STATUS_SUCCESS) {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

/* CUDA parallel sssp */
void cuda_sssp(int * graph_mat, int src_node, long * node_distance_array) {
  /* Initialization */
  for (size_t i = 0; i < NODES; i++) {
    node_distance_array[i] = INFI;
  }

  /* Convert the graph matrix to the COO-format */
  int num_edges = 0;
  for (size_t i = 0; i < NODES; i++) {
    for (size_t j = 0; j < NODES; j++) {
      if (graph_mat[i * NODES + j] != 0) {
        num_edges++;
      }
    }
  }

  int * source_indices = (int *) malloc (num_edges * sizeof(int));
  int * destination_indices = (int *) malloc (num_edges * sizeof(int));
  float * weights = (float *) malloc (num_edges * sizeof(float));

  int edge_index = 0;
  for (size_t i = 0; i < NODES; i++) {
    for (size_t j = 0; j < NODES; j++) {
      if (graph_mat[i * NODES + j] != 0) {
        source_indices[edge_index] = i;
        destination_indices[edge_index] = j;
        weights[edge_index] = (float) graph_mat[i * NODES + j];
        edge_index++;
      }
    }
  }

  nvgraphCOOTopology32I_st coo_topology;
  float * coo_weights;
  cudaMalloc(&coo_topology.source_indices, num_edges*sizeof(int));
  cudaMalloc(&coo_topology.destination_indices, num_edges*sizeof(int));
  cudaMalloc(&coo_weights, num_edges*sizeof(long));

  coo_topology.nvertices = NODES;
  coo_topology.nedges = num_edges;
  cudaMemcpy(coo_topology.source_indices, source_indices, num_edges*sizeof(int), cudaMemcpyDefault);
  cudaMemcpy(coo_topology.destination_indices, destination_indices, num_edges*sizeof(int), cudaMemcpyDefault);
  coo_topology.tag = NVGRAPH_UNSORTED;
  cudaMemcpy(coo_weights, weights, num_edges*sizeof(long), cudaMemcpyDefault);

  /*
  Start to run nvgraphSssp(), the following codes are modified from nvGRAPH SSSP example
  (https://docs.nvidia.com/cuda/nvgraph/index.html#nvgraph-sssp-example)
  */
  const size_t vertex_numsets = 1, edge_numsets = 1;
  float * sssp_1_h;
  void ** vertex_dim;
  nvgraphHandle_t handle;
  nvgraphGraphDescr_t graph;
  nvgraphCSCTopology32I_st csc_topology;
  cudaDataType_t edge_dimT = CUDA_R_32F;
  cudaDataType_t* vertex_dimT;
  sssp_1_h = (float *) malloc(NODES * sizeof(float));
  vertex_dim  = (void **) malloc(vertex_numsets * sizeof(void*));
  vertex_dimT = (cudaDataType_t*) malloc(vertex_numsets * sizeof(cudaDataType_t));
  float * csc_weights;
  cudaMalloc(&csc_topology.source_indices, num_edges * sizeof(int));
  cudaMalloc(&csc_topology.destination_offsets, (NODES+1) * sizeof(int));
  cudaMalloc(&csc_weights, num_edges * sizeof(float));
  vertex_dim[0]= (void *) sssp_1_h; vertex_dimT[0] = CUDA_R_32F;

  check(nvgraphCreate(&handle));
  check(nvgraphCreateGraphDescr (handle, &graph));
  nvgraphConvertTopology(handle,
                         NVGRAPH_COO_32, &coo_topology, coo_weights,
                         &edge_dimT,
                         NVGRAPH_CSC_32, &csc_topology, csc_weights);

  check(nvgraphSetGraphStructure(handle, graph, (void *) &csc_topology, NVGRAPH_CSC_32));
  check(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
  check(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
  check(nvgraphSetEdgeData(handle, graph, (void *) csc_weights, 0));

  struct timeval t1, t2;
  gettimeofday(&t1, NULL);
  int source_vert = src_node;
  check(nvgraphSssp(handle, graph, 0,  &source_vert, 0));
  check(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));
  for (int i = 0; i < NODES; i++) {
    node_distance_array[i] = (long) sssp_1_h[i];
  }
  gettimeofday(&t2, NULL);
  double used_time = t2.tv_sec-t1.tv_sec +(t2.tv_usec-t1.tv_usec)/1000000.0;
  printf("CUDA sssp (algorithm running time only) used time: %f sec\n", used_time);

  /* Free resources */
  free(source_indices);
  free(destination_indices);
  free(weights);
  free(sssp_1_h);
  free(vertex_dim);
  free(vertex_dimT);
  cudaFree(csc_topology.source_indices);
  cudaFree(csc_topology.destination_offsets);
  cudaFree(csc_weights);
  cudaFree(coo_topology.source_indices);
  cudaFree(coo_topology.destination_indices);
  cudaFree(coo_weights);
  check(nvgraphDestroyGraphDescr(handle, graph));
  check(nvgraphDestroy(handle));
}
/*************************** nvGraph CUDA parallel implementation for sssp ***************************/
