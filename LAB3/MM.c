#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>

const int N = 4;
int *local_coords;

void creat_ring(MPI_Comm *comm1, int *rank, int *world_size);
void creat_2dmesh(MPI_Comm *comm2, int *rank, int *world_size);

float *generate_matrix(int world_size, int rank);
float *generate_ring(MPI_Comm *comm1, int rank, int world_size, int *N_per_node);
float *generate_mesh(MPI_Comm *comm2, int rank, int world_size, int *N_per_node);

float Det(float *local_A, int N);
float *Mat_Mul(float *local_A, float *local_B, int N);
float *Cannon_Mat_Mul(float *local_A, float *local_B, int N, MPI_Comm comm2);
float *Matrix_Chain(int N, int k);
float *Mult(float *local_A, float *s, int i, int j, int N);


int main (int argc, char **argv)
{
   int world_size;
   int rank;
   int i;
   int k = 4;
   int K = 4;
   int N_per_node;
   float *local_A, *local_B, *local_MM, *s;
   float det[1], result[1];
   double ring_time, mesh_time, ring_cannon_time, mesh_cannon_time, ring_chain_time, mesh_chain_time;
   
   MPI_Status status;
   MPI_Comm comm1;
   MPI_Comm comm2;
   
   MPI_Init(&argc, &argv); //Initialize MPI environment 
   MPI_Comm_size(MPI_COMM_WORLD, &world_size); //get the total number of processor
   
   printf("The total number of processor: %d\n", world_size);
   printf("\nThe size of the matrix is: %d x %d\n", N, N);
   printf("\nThe value of k is: %d\n", k);
   
   //ring mapping and calculation
   printf("Starting the ring mapping and calculation\n");


   creat_ring(&comm1, &rank, &world_size);
   ring_time = MPI_Wtime(); 
   local_A = generate_ring(&comm1, rank, world_size, &N_per_node);  
   
   //performing traditional (n^3) algorithm
   if (k=1) //if k=1, find the determinant directly
   {
   	 det [0] = Det(local_A, N);
	 MPI_Reduce(&det, &result, 1, MPI_FLOAT, MPI_PROD, 0, comm1);
	 if (rank == 0)
	 {
	 	//printf("\nThe determinant of the matrix using ring mapping and n^3 algorithm is: %f\n", result[0]);
	 }
   }
   else //if k>1, perfom the matrix multiplication then find the determinant
   {
    local_B = local_A;
    for (i=0; i<k; i++)
	{
	    local_MM = Mat_Mul(local_A, local_B, N);
	}
   	det [0] = Det(local_MM, N);
	MPI_Reduce(&det, &result, 1, MPI_FLOAT, MPI_PROD, 0, comm1);
	if (rank == 0)
	{
	 printf("\nThe determinant of the matrix using ring mapping and n^3 algorithm is: %f\n", result[0]);
	}
   }
   
   ring_time = MPI_Wtime() - ring_time; //measure the total execuation time for the ring mapping
   printf("\nThe total time is needed to find the determinant using ring mapping and n^3 algorithm: %f\n", ring_time);
   
   //performing cannon's method
   ring_cannon_time = MPI_Wtime(); 
   if (K>1)
   {
    local_A = generate_ring(&comm1, rank, world_size, &N_per_node);
   	local_B = local_A;
	for (i=0; i<k; i++)
	{
	    local_MM = Cannon_Mat_Mul(local_A, local_B, N, comm1);
	}
   	det [0] = Det(local_MM, N);
	MPI_Reduce(&det, &result, 1, MPI_FLOAT, MPI_PROD, 0, comm1);
	if (rank == 0)
	{
	 printf("\nThe determinant of the matrix using ring mapping and cannon's method is: %f\n", result[0]);
	}
   }
   ring_cannon_time = MPI_Wtime() - ring_cannon_time; //measure the total execuation time for the cannon's method
   printf("\nThe total time is needed to find the determinant using ring mapping and cannon's method: %f\n", ring_cannon_time);
 
   //performing chain matrix multiplication 
   ring_chain_time = MPI_Wtime(); 
   if (K>1)
   {
    local_A = generate_mesh(&comm1, rank, world_size, &N_per_node);
	for (i=0; i<K; i++)
	{
		s = Matrix_Chain(N, k);
		local_MM = Mult(local_A, s, N, N, N);
	}
   	det [0] = Det(local_MM, N);
	MPI_Reduce(&det, &result, 1, MPI_FLOAT, MPI_PROD, 0, comm1);
	if (rank == 0)
	{
	 printf("\nThe determinant of the matrix using ring mapping and Matrix Chain algorithm is: %f\n", result[0]);
	}
   }
   ring_chain_time = MPI_Wtime() - ring_chain_time; //measure the total execuation time for the chain matrix multiplication
   printf("\nThe total time is needed to find the determinant using ring mapping and Matrix Chain algorithm: %f\n", ring_chain_time);
   
   
   //2D mesh mapping and calculation 
   printf("Staring the 2D mesh mapping and calculation\n");
   
   creat_2dmesh(&comm2, &rank, &world_size);
   N_per_node = N / sqrt(world_size);
   mesh_time = MPI_Wtime();
   local_A = generate_mesh(&comm2, rank, world_size, &N_per_node);
   
   //performing traditional (n^3) algorithm
   if (k=1) //if k=1, find the reterminant directly
   {
   	 det [0] = Det(local_A, N_per_node);
	 MPI_Reduce(&det, &result, 1, MPI_FLOAT, MPI_PROD, 0, comm2);
	 if (rank == 0)
	 {
	 	//printf("\nThe determinant of the matrix using 2D mesh mapping and n^3 algorithm is: %f\n", result[0]);
	 }
   }
   else //if k>1, berfom the matrix multiplication then find the determinant
   {
    local_B = local_A;
    for (i=0; i<k; i++)
	{
	    local_MM = Mat_Mul(local_A, local_B, N_per_node);
	}
   	det [0] = Det(local_MM, N_per_node);
	MPI_Reduce(&det, &result, 1, MPI_FLOAT, MPI_PROD, 0, comm2);
	if (rank == 0)
	{
	 printf("\nThe determinant of the matrix using 2D mesh mapping and n^3 algorithm is: %f\n", result[0]);
	}
   }
   mesh_time = MPI_Wtime() - mesh_time; //measure the total execuation time for the mesh mapping
   printf("\nThe total time is needed to find the determinant using 2D mesh mapping and n^3 algorithm: %f\n", mesh_time);
   
   //performing cannon's method
   mesh_cannon_time = MPI_Wtime(); 
   if (K>1)
   {
    local_A = generate_mesh(&comm2, rank, world_size, &N_per_node);
   	local_B = local_A;
	for (i=0; i<k; i++)
	{
	    local_MM = Cannon_Mat_Mul(local_A, local_B, N_per_node, comm2);
	}
   	det [0] = Det(local_MM, N_per_node);
	MPI_Reduce(&det, &result, 1, MPI_FLOAT, MPI_PROD, 0, comm2);
	if (rank == 0)
	{
	 printf("\nThe determinant of the matrix using 2D mesh mapping and cannon's method is: %f\n", result[0]);
	}
   }
   mesh_cannon_time = MPI_Wtime() - mesh_cannon_time; //measure the total execuation time for the cannon's method
   printf("\nThe total time is needed to find the determinant using 2D mesh mapping and cannon's method: %f\n", mesh_cannon_time);
   
   //performing chain matrix multiplication 
   mesh_chain_time = MPI_Wtime(); 
   if (K>1)
   {
    local_A = generate_mesh(&comm2, rank, world_size, &N_per_node);
	for (i=0; i<K; i++)
	{
		s = Matrix_Chain(N_per_node, k);
		local_MM = Mult(local_A, s, N_per_node, N_per_node, N);
	}
   	det [0] = Det(local_MM, N_per_node);
	MPI_Reduce(&det, &result, 1, MPI_FLOAT, MPI_PROD, 0, comm2);
	if (rank == 0)
	{
	 printf("\nThe determinant of the matrix using 2D mesh mapping and Matrix Chain algorithm is: %f\n", result[0]);
	}
   }
   
   
   mesh_chain_time = MPI_Wtime() - mesh_chain_time; //measure the total execuation time for the chain matrix multiplication
   printf("\nThe total time is needed to find the determinant using 2D mesh mapping and Matrix Chain algorithm: %f\n", mesh_chain_time);
  
   free(local_A);
   MPI_Comm_free(&comm1);
   MPI_Comm_free(&comm2);
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

//create the 2d-mesh topolgy
void creat_2dmesh(MPI_Comm *comm2, int *rank, int *world_size)
{
   MPI_Comm_size(MPI_COMM_WORLD, world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, rank);
	
   int i;
   int dims[2];
   int periods[2];
   int p = (int) sqrt((double) *world_size);
   local_coords = (int*) malloc(sizeof(int) * 2);
   
   for (i=0; i<2; i++)
   {
   	dims[i] = p;
	periods[i] = 0;
   }
   MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, comm2);
   MPI_Comm_size(*comm2, world_size);
   MPI_Cart_coords(*comm2, *rank, 2, local_coords);
}

//Initialize the matrix randomly with the values {-1, 0 , 1}  
float *generate_matrix(int world_size, int rank)
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
		}
	}
	Ts = MPI_Wtime() - Ts;
	printf("\nSerial time = %d\n", Ts);
	return gen_matrix;
}

//send the matrix over the ring topology
float *generate_ring(MPI_Comm *comm1, int rank, int world_size, int *N_per_node)
{
	float *local_A;
	int i, j;
	
	if (rank == 0)
	{
		local_A = generate_matrix(world_size, rank);
	}
	else
	{
		local_A = (float*) malloc(sizeof(float) * N * N);
	}
	MPI_Bcast(local_A, N*N, MPI_FLOAT, 0, *comm1);
	return local_A;
}

//send the matrix over the 2d-mesh topology
float *generate_mesh(MPI_Comm *comm2, int rank, int world_size, int *N_per_node)
{
	float *local_A;
	float *temp_A;
	float *buff_to_send;
	int i, j, k, l;
	int p = (int) sqrt(world_size);
	int n_proc = N / p;
	MPI_Status status;
	
	local_A = (float*) malloc(sizeof(float) * n_proc * n_proc);
	if (rank == 0)
	{
	 temp_A = generate_matrix(world_size, rank);
	 
	 for (i=0; i<p; i++)
	 {
	 	for (j=0; j<p; j++)
		{
			if (i==0 && j==0)
			{
				int index = 0;
				for (k=0; k<n_proc; k++)
				{
					for (l=0; l<n_proc; l++)
					{
						local_A[index++] = temp_A[k*N+1];
					}
				}
			}
			else
			{
				buff_to_send = (float*) malloc(sizeof(float) * (n_proc * n_proc));
				int S_row = i * n_proc;
				int S_col = j * n_proc;
				int index = 0;
		
				for (k=S_row; k<S_row+n_proc; k++)
				{
					for (l=S_col; l<S_col+n_proc; l++)
					{
						buff_to_send[index++] = temp_A[k*N+1];
					}
				}
				MPI_Send(buff_to_send, n_proc*n_proc, MPI_FLOAT, j*p+i,0, *comm2);
				free (buff_to_send);
			}	
		}
	 }
	 free (temp_A);	 
	}
	else 
	{
	 MPI_Recv(local_A, n_proc*n_proc, MPI_FLOAT,0,0, *comm2, &status);
	}	
	return local_A;
}

//Calulate the determinant of the matrix
float Det(float *local_A, int N)
{
	int i, j, k;
	float determinant = 0;
	
	for (k=1; k<N; k++)
	{
		for (i=k+1; i<=N; i++)
		{
			for (j=k+1; j<=N; j++)
			{
				determinant = local_A[i*N+j] * local_A[k*N+k] - local_A[i*N+k] * local_A[k*N+j];
				if (k >= 2)
				{
				  determinant = local_A[i*N+j] / local_A[k*N+k];
				}
			}
		}
	}
	return determinant; 
}

//Perform matrix multiplication using traditional n^3 algorithm 
float *Mat_Mul(float *local_A, float *local_B, int N)
{
	int i, j, k;
	float *local_MM;
	local_MM = (float*) malloc(sizeof(float) * (N*N));
	
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			local_MM[i*N+j] = 0;
			for (k=0; k<N; k++)
			{
				local_MM[i*N+j] = local_MM[i*N+j] + local_A[i*N+k] * local_B[k*N+j];
			}
		}
	}
	return local_MM;
}

//Perform matrix multiplication using Cannon's method
float *Cannon_Mat_Mul(float *local_A, float *local_B, int N, MPI_Comm comm)
{
	int i;
	int nlocal, npes;
	int mycoords[2], dims[2], periods[2];
	int myrank, my2drank, rightrank, leftrank, downrank, uprank;
	int shiftsource, shiftdest;
	float *local_MM;
	local_MM = (float*) malloc(sizeof(float) * (N*N));
	
	MPI_Status status; 
	MPI_Comm comm_2d;
	
	MPI_Comm_size(comm, &npes);
	MPI_Comm_rank(comm, &myrank);
	
	dims[0] = dims[1] = sqrt(npes);
	periods[0] = periods[1] = 1; //for wraparound connenctions
	
	MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d);
	
	//get the rank and coordinate 
	MPI_Comm_rank(comm_2d, &my2drank);
	MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);
	
	//compute the rank of the up and left shift
	MPI_Cart_shift(comm_2d, 0, -1, &rightrank, &leftrank);
	MPI_Cart_shift(comm_2d, 1, -1, &downrank, &uprank);
	
	nlocal = N/dims[0]; //determine the dimension of the local matrix block
	
	//perform the initial matrix alignment
	MPI_Cart_shift(comm_2d, 0, -1, &shiftsource, &shiftdest);
	MPI_Sendrecv_replace(local_A, nlocal*nlocal, MPI_FLOAT, shiftdest, 1, shiftsource, 1, comm_2d, &status);
	MPI_Cart_shift(comm_2d, 1, -1, &shiftsource, &shiftdest);
	MPI_Sendrecv_replace(local_B, nlocal*nlocal, MPI_FLOAT, shiftdest, 1, shiftsource, 1, comm_2d, &status);
	
	//get into the main computaion loop
	for (i=0; i<dims[0]; i++)
	{
		local_MM = Mat_Mul(local_A, local_B, nlocal);
		//shift matrix local_A left by one
		MPI_Sendrecv_replace(local_A, nlocal*nlocal, MPI_FLOAT, leftrank, 1, rightrank, 1, comm_2d, &status);
		//shift matrix local_B up by one
		MPI_Sendrecv_replace(local_B, nlocal*nlocal, MPI_FLOAT, uprank, 1, downrank, 1, comm_2d, &status);
	}
	
	//restore the original distribuation os local_A and local_B
	MPI_Cart_shift(comm_2d, 0, 1, &shiftsource, &shiftdest);
	MPI_Sendrecv_replace(local_A, nlocal*nlocal, MPI_FLOAT, shiftdest, 1, shiftsource, 1, comm_2d, &status);
	MPI_Cart_shift(comm_2d, 1, 1, &shiftsource, &shiftdest);
	MPI_Sendrecv_replace(local_B, nlocal*nlocal, MPI_FLOAT, shiftdest, 1, shiftsource, 1, comm_2d, &status);
	
	MPI_Comm_free(&comm_2d);
	return local_MM;
}

//perform chain matrix multiplication (Dynamic Programing Algorithm)
float *Matrix_Chain(int N, int k)
{
	int i, j, l, m;
	float q;
	float *M;
	float *s;
	
	M = (float*) malloc(sizeof(float) * (N*N));
	s = (float*) malloc(sizeof(float) * (N*N));
	
	for (i=1; i<k; i++)
	{
		M[i*N+i] = 0; //initialize
	}
	for (l=2; l<k; l++) //l = length of subchain
	{
		for (i=1; i<k-l+1; i++)
		{
			j = i + l -1;
			M[i*N+j] = INFINITY;
			for (m=i; m<j-1; m++)	//check all splits
			{
				q = M[i*N+m] + M[m+1*N+j] + N * N * N;
				if (q < M[i*N+j])
				{
					M[i*N+j] = q;
					s[i*N+j] =m;
				}
			}
		}
	}
	return M, s;
}
float *Mult(float *local_A, float *s, int i, int j, int N)
{
	float *X, *Y;
	X = (float*) malloc(sizeof(float) * (N*N));
	Y = (float*) malloc(sizeof(float) * (N*N));
	
	if (i < j)
	{
		X = Mult(local_A, s, i, s[i*N+j], N);
		Y = Mult(local_A, s, s[i*N+j]+1, j, N);
	}
	else
	return &local_A[i*N+0];
}
