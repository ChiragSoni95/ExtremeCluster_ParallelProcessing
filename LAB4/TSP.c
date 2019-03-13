#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

 //MPI program to solve the Travelling Salesman Problem  using parallel branch-and-bound tree search algorithm 
int *local_coords;
FILE *myfile;

void creat_ring(MPI_Comm *comm1, int *rank, int *world_size);

float *generate_matrix(int world_size, int rank, int N);
float *generate_ring(MPI_Comm *comm1, int rank, int world_size, int *N_per_node, int N);

int main (int argc, char **argv)
{
   int world_size;
   int rank;
   int i,x;
   int N;
   int N_per_node;
   float *local_A;
   float det[1], result[1];
   double ring_time, mesh_time, ring_cannon_time, mesh_cannon_time;
   
   MPI_Status status;
   MPI_Comm comm1;
   
   myfile = fopen("data.txt", "w");
   if (myfile == NULL)
   {
   	fprintf(stderr, "failed to open data.txt\n");
	exit(1);
   }
   char line[280]; // 280 pair of co-ordinates
   fgets(line, 280, myfile); //discard the first line
   fgets(line, 280, myfile); //discard 2nd line
   fgets(line, 280, myfile); //discard 3rd line
   fgets(line, 280, myfile); //get the line for dimension
   line[strcspn(line, "\r\n")] = '\0'; //strip EPL(s) char at end
   char *token;
   token = strtok(line, " ");
   token = strtok(line, " ");
   token = strtok(line, " ");
   N = atoi(token); //store the number of cities to N
   N=280;
   fgets(line, 280, myfile); //discard the first line
   fgets(line, 280, myfile); //discard 2nd line
   
   MPI_Init(&argc, &argv); //Initialize MPI environment 
   MPI_Comm_size(MPI_COMM_WORLD, &world_size); //get the total number of processor
   
   printf("The total number of processor: %d\n", world_size);
   printf("\nThe size of the matrix is: %d x %d\n", N, N);
   
   //ring mapping and calculation
   printf("Staring the ring mapping and calculation\n");

   creat_ring(&comm1, &rank, &world_size);
   ring_time = MPI_Wtime(); 
   local_A = generate_ring(&comm1, rank, world_size, &N_per_node, N);  
 
   ring_time = MPI_Wtime() - ring_time; //measure the total execuation time for the ring mapping
   printf("\nThe total time to calclulate the shortest path is: %f\n", ring_time);
    
   free(local_A);
   MPI_Comm_free(&comm1);
   MPI_Finalize(); //Finalizing MPI
  
   return 0; 
} 

//create the ring topology
void creat_ring(MPI_Comm *comm1, int *rank, int *world_size)
{	
	MPI_Comm_size(MPI_COMM_WORLD, world_size);
	
	int dims[1], periods[1];
	dims[0] = *world_size;
	periods[0] = 1;
	local_coords = (int*) malloc(sizeof(int) * 1);
	
	MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, comm1);
	MPI_Comm_rank(*comm1, rank);
	MPI_Comm_size(*comm1, world_size);
	MPI_Cart_coords(*comm1, *rank, 1, local_coords);
}


//Initialize the matrix randomly with the values {-1, 0 , 1}  
float *generate_matrix(int world_size, int rank, int N)
{
	int i, j;
	float *gen_matrix;
	int low = -1;
	int high = 1;
	double Ts;
	
	Ts = MPI_Wtime();
	gen_matrix = (float*) malloc(sizeof(float) * (N*N));
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			gen_matrix[i*N+j] = (int)(rand() % (high - low + 1) + low);
			//printf("%f\t", gen_matrix[i*N+j]);
		}
		//printf("\n");
	}
	Ts = MPI_Wtime() - Ts;
	printf("\nStartup time = %d\n", Ts);
	return gen_matrix;
}

//send the matrix over the ring topology
float *generate_ring(MPI_Comm *comm1, int rank, int world_size, int *N_per_node, int N)
{
	float *local_A;
	int i, j;
	
	if (rank == 0)
	{
		local_A = generate_matrix(world_size, rank, N);
	}
	else
	{
		local_A = (float*) malloc(sizeof(float) * N * N);
	}
	MPI_Bcast(local_A, N*N, MPI_FLOAT, 0, *comm1);
	return local_A;
}
