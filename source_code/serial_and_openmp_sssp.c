#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <limits.h>
#include <omp.h>

#define NODES 40000       //number of nodes
#define MIN_DEGREE 512    //minimum number of edges per node
#define INFI INT_MAX      //infinity value for distance
#define THREAD_NUM 4      //number of threads for OpenMP

void graph_initialization(int * graph, int * edge_degree_array);
void print_graph_info(int * graph);
void serial_sssp(int * graph, int src_node, long * node_distance_array, int * parent_nodes_array, int * visited_nodes_array);
void openmp_sssp(int * graph, int src_node, long * node_distance_array, int * parent_nodes_array, int * visited_nodes_array);


int main(int argc, char const *argv[]) {

  /* Setup necessary arrays and matrices */
  int * graph = (int *) malloc (NODES * NODES * sizeof(int));
  long * node_distance_array = (long *) malloc (NODES * sizeof(long));
  int * parent_nodes_array = (int *) malloc (NODES * sizeof(int));
  int * edge_degree_array = (int *) malloc (NODES * sizeof(int));
  int * visited_nodes_array = (int *) malloc (NODES * sizeof(int));

  /* One row for serial_sssp, One row for openmp_sssp */
  int * parent_nodes_results_matrix = (int *) malloc (2 * NODES * sizeof(int));
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
  /* Copy results to the result matrices for later result comparison */
  for (size_t i = 0; i < NODES; i++) {
    parent_nodes_results_matrix[0 * NODES + i] =parent_nodes_array[i];
    distance_results_matrix[0 * NODES + i] = node_distance_array[i];
  }

  /* Start to run OpenMP parallel sssp */
  omp_set_num_threads(THREAD_NUM);
  gettimeofday(&t1, NULL);
  openmp_sssp(graph, src_node, node_distance_array, parent_nodes_array, visited_nodes_array);
  gettimeofday(&t2, NULL);
  used_time = t2.tv_sec-t1.tv_sec +(t2.tv_usec-t1.tv_usec)/1000000.0;
  printf("OpenMP sssp used time: %f sec\n", used_time);
  /* Copy results to the result matrices for later result comparison */
  for (size_t i = 0; i < NODES; i++) {
    parent_nodes_results_matrix[1 * NODES + i] =parent_nodes_array[i];
    distance_results_matrix[1 * NODES + i] = node_distance_array[i];
  }

  /* For debug */
  // for (size_t i = 0; i < NODES; i++) {
  //   printf("%*d ", 6, parent_nodes_results_matrix[i]);
  // }
  // printf("\n");
  // for (size_t i = 0; i < NODES; i++) {
  //   printf("%*d ", 6, parent_nodes_results_matrix[NODES + i]);
  // }
  // printf("\n");
  // printf("===============================================\n");
  // for (size_t i = 0; i < NODES; i++) {
  //   printf("%*ld ", 6, distance_results_matrix[i]);
  // }
  // printf("\n");
  // for (size_t i = 0; i < NODES; i++) {
  //   printf("%*ld ", 6, distance_results_matrix[NODES + i]);
  // }
  // printf("\n");

  /* Compare the results */
  int pn_errors = 0;
  int dis_errors = 0;
  for (size_t i = 0; i < NODES; i++) {
    if (parent_nodes_results_matrix[i] != parent_nodes_results_matrix[NODES + i])
      pn_errors += 1;
    if (distance_results_matrix[i] != distance_results_matrix[NODES + i])
      dis_errors += 1;
  }
  printf("pn_errors = %d, dis_errors = %d\n", pn_errors, dis_errors);

  /* free all resources */
  free(graph);
  free(node_distance_array);
  free(parent_nodes_array);
  free(edge_degree_array);
  free(visited_nodes_array);
  free(parent_nodes_results_matrix);
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


/*************************** OpenMP parallel implementation for sssp ***************************/
/* extract_min using OpenMP */
int openmp_extract_min(long * node_distance_array, int * visited_nodes_array) {
  int node = 0;
  int dist = INFI;

  #pragma omp parallel shared(node_distance_array, visited_nodes_array)
  {
      int tmp_node = node;
      int tmp_dist = dist;
      #pragma omp barrier //ensure each thread starts with same data

      #pragma omp for
      for (size_t i = 0; i < NODES; i++) {
        if (node_distance_array[i] < tmp_dist && visited_nodes_array[i] == 0) {
          tmp_node = i;
          tmp_dist = node_distance_array[i];
        }
      }

      #pragma omp critical
      {
        if (tmp_dist < dist) {
          node = tmp_node;
          dist = tmp_dist;
        }
      }
  }
  return node;
}


/* OpenMP parallel sssp */
void openmp_sssp(int * graph, int src_node, long * node_distance_array, int * parent_nodes_array, int * visited_nodes_array) {
  /* Initialization */
  for (size_t i = 0; i < NODES; i++) {
    node_distance_array[i] = INFI;
    parent_nodes_array[i] = -1;
    visited_nodes_array[i] = 0;
  }
  node_distance_array[src_node] = 0;

  /* Dijkstra Core */
  for (size_t i = 0; i < NODES; i++) {
    int cur_node = openmp_extract_min(node_distance_array, visited_nodes_array);
    visited_nodes_array[cur_node] = 1;

    #pragma omp parallel for shared(graph, node_distance_array)
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
    #pragma omp barrier
  }
}
/*************************** OpenMP parallel implementation for sssp ***************************/
