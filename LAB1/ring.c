#include <mpi.h>
#include <stdio.h>
#include <math.h>
int main(int argc, char **argv) {
   int i, j;
   int start, end;    
   int sum = 0;
   int ownSum=0;
   MPI_Status status; /* return status for receive*/  
   int world_size;
   int rank;
   char hostname[256];
   char processor_name[MPI_MAX_PROCESSOR_NAME];
   int name_len;
   int master = 0;
   int v=0;
   int tag = 0;
   int N;
   int factor=0;
   int t=0;
   const int size=1000; 
   int array[size];
   double final_comp=0;
   double t1_comm,t2_comm,comm_time;
   double t1_comp,t2_comp,comp_time;
/*Startup MPI*/
  MPI_Init(&argc,&argv);

 /*Find out number of processes*/
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

/*Find out processes rank */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

 /*Get the processor name*/
  MPI_Get_processor_name(processor_name, &name_len); //processor_name must be of size atleast MPI_MAX_PROCESSOR_NAME

  gethostname(hostname,255);
   
   
/*Create an aray by random number generation*/
   N=size/world_size;
   factor=N*rank;
     for(i = 0;i<size;i++){
	array[i] = rand() % 10;
  }	

/*Calculating the sum of each chunk depending on the total number of processors and size of the array*/
   for (i=0;i<N;i++)
   	{
   		ownSum=ownSum+array[i+factor];
	 }

/*Check if the rank of the processor is 0 i.e if it is the source processor*/
   if(rank==0)
   {
       
	   /*Start time where the processors start communicating and sending message*/
	   t1_comm=MPI_Wtime();
	   MPI_Send(&sum,1,MPI_INT,(rank+1),tag,MPI_COMM_WORLD);
	   MPI_Recv(&sum,1,MPI_INT,world_size-1,0,MPI_COMM_WORLD,&status);
	   /*End time where the final message from the processor with the last rank reaches the first processor i.e processor 0*/
	   t2_comm=MPI_Wtime();
           comm_time=t2_comm-t1_comm;
	   sum=sum+ownSum;
	   /*Computation ends when the sum of the final chunks get added to the resultant final sum*/
	   t2_comp=MPI_Wtime();
	   comp_time=t2_comp-t1_comp;
	   printf("The sum of the numbers in array: %d\n",sum);
	   printf("The time for computation is %lf seconds\n",comp_time);
	   printf("The time for communication is %lf picoseconds\n",comm_time);
	   
   }

/*Check if the rank of the processor is between P0 and P(n-1)*/
   if((rank>0)&&(rank<world_size-1))
   {
   	MPI_Recv(&sum,1,MPI_INT,rank-1,0,MPI_COMM_WORLD,&status);
	/*Computation starts when the sum of the first chunk is added to the total sum*/
   	t1_comp=MPI_Wtime();
	sum=sum+ownSum;
   	MPI_Send(&sum,1,MPI_INT,(rank+1),tag,MPI_COMM_WORLD);
   }

/*Check if the rank of the processor is (n-1) i.e if it the last processor and now it will send the computed sum back to P0*/
   if(rank==world_size-1)
   {
   	MPI_Recv(&sum,1,MPI_INT,rank-1,0,MPI_COMM_WORLD,&status);
   	sum=sum+ownSum;
   	MPI_Send(&sum,1,MPI_INT,0,tag,MPI_COMM_WORLD);
   	} 
   
  MPI_Finalize();

  return 0;
}
